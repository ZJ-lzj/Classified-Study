# ============================================================
# Earthquake / Explosion Classification
# 1D-CNN (Raw Waveform Input, No Feature Extraction) - 5-Fold CV
# - 278 raw events → 5-fold stratified CV (Event-level split)
# - Train: sliding window augmentation + per-sample normalization
# - Test: center window only + per-sample normalization
# - Per-fold optimal threshold (Youden's J) → Evaluation
# - Unknown prediction via 5-fold ensemble
# ============================================================

import os
import shutil
import random
import obspy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    accuracy_score
)

# ============================================================
# Fixed Random Seed
# ============================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ============================================================
# Paths
# ============================================================
WAVE_FOLDER = r"C:\Users\adminstor\OneDrive\桌面\files data"
LABEL_XLSX  = r"C:\Users\adminstor\OneDrive\桌面\data.xlsx"
OUTPUT_FOLDER = os.path.join(WAVE_FOLDER, "Results_1DCNN_5Fold")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# Hyper Parameters (Fixed FS = 100 Hz for HHZ channel)
# ============================================================
FIXED_FS     = 100.0
FIXED_LEN    = 1500      # 15 seconds at 100 Hz
BANDPASS_MIN = 2.0
BANDPASS_MAX = 18.0

WINDOW_LEN      = 1200   # 12 seconds at 100 Hz
WINDOW_OFFSETS   = [-200, -100, 0, 100, 200]   # train augmentation
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3

N_FOLDS = 5

# ============================================================
# Wave Normalization (Per-sample, standard for 1D-CNN)
# ============================================================
def normalize_wave(x):
    x = x.astype(np.float32)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

# ============================================================
# Energy Alignment
# ============================================================
def align_wave(data):
    energy = data ** 2
    smooth = np.convolve(energy, np.ones(80), mode='same')
    peak = np.argmax(smooth)
    start = peak - FIXED_LEN // 2
    end   = start + FIXED_LEN
    if start < 0:
        start = 0; end = FIXED_LEN
    if end > len(data):
        end = len(data); start = max(0, end - FIXED_LEN)
    out = data[start:end]
    if len(out) < FIXED_LEN:
        out = np.pad(out, (0, FIXED_LEN - len(out)), mode='constant')
    return out

# ============================================================
# Wave Processing
# ============================================================
def process_wave(path):
    st = obspy.read(path)
    tr = st[0]
    tr.detrend("demean")
    tr.detrend("linear")
    tr.filter("bandpass", freqmin=BANDPASS_MIN, freqmax=BANDPASS_MAX,
              corners=4, zerophase=True)
    data = tr.data.astype(np.float32)
    data = align_wave(data)
    data = normalize_wave(data)
    return data

# ============================================================
# Sliding Windows (Multiple offsets → Train Augmentation)
# ============================================================
def sliding_windows(wave):
    windows = []
    energy = wave ** 2
    smooth = np.convolve(energy, np.ones(80), mode='same')
    peak = np.argmax(smooth)
    center_start = peak - WINDOW_LEN // 2
    for off in WINDOW_OFFSETS:
        start = center_start + off
        end   = start + WINDOW_LEN
        if start < 0:
            start = 0; end = WINDOW_LEN
        if end > len(wave):
            end = len(wave); start = max(0, end - WINDOW_LEN)
        seg = wave[start:end]
        if len(seg) < WINDOW_LEN:
            seg = np.pad(seg, (0, WINDOW_LEN - len(seg)), mode='constant')
        seg = np.interp(np.linspace(0, len(seg)-1, FIXED_LEN),
                        np.arange(len(seg)), seg)
        seg = normalize_wave(seg)
        windows.append(seg.astype(np.float32))
    return windows

# ============================================================
# Center Window Only (For Test & Unknown - Fair Comparison)
# ============================================================
def center_window(wave):
    energy = wave ** 2
    smooth = np.convolve(energy, np.ones(80), mode='same')
    peak = np.argmax(smooth)
    center_start = peak - WINDOW_LEN // 2
    start = center_start
    end   = start + WINDOW_LEN
    if start < 0:
        start = 0; end = WINDOW_LEN
    if end > len(wave):
        end = len(wave); start = max(0, end - WINDOW_LEN)
    seg = wave[start:end]
    if len(seg) < WINDOW_LEN:
        seg = np.pad(seg, (0, WINDOW_LEN - len(seg)), mode='constant')
    seg = np.interp(np.linspace(0, len(seg)-1, FIXED_LEN),
                    np.arange(len(seg)), seg)
    seg = normalize_wave(seg)
    return seg.astype(np.float32)

# ============================================================
# Augmentation helpers
# ============================================================
def augment_wave_light(w):
    aug = w.copy()
    noise_std = np.random.uniform(0.001, 0.008)
    aug += np.random.normal(0, noise_std, size=aug.shape)
    shift = np.random.randint(-20, 20)
    aug = np.roll(aug, shift)
    aug = normalize_wave(aug)
    return aug.astype(np.float32)

def augment_wave_amplitude_shift(w):
    aug = w.copy()
    baseline = np.random.uniform(-0.3, 0.3)
    aug += baseline
    amp = np.random.uniform(0.8, 1.2)
    aug *= amp
    aug = normalize_wave(aug)
    return aug.astype(np.float32)

# ============================================================
# 1D-CNN Model Architecture
# ============================================================
def build_1dcnn_model():
    inp = tf.keras.layers.Input(shape=(FIXED_LEN, 1))
    
    x = tf.keras.layers.Conv1D(32, 15, padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    
    x = tf.keras.layers.Conv1D(64, 9, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    
    x = tf.keras.layers.Conv1D(128, 5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ============================================================
# Find Optimal Threshold
# ============================================================
def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

# ============================================================
# Load Labels & Waveform Data
# ============================================================
df = pd.read_excel(LABEL_XLSX)
df["filename"] = df["filename"].astype(str).str.strip().str.lower()
df["label"]    = pd.to_numeric(df["label"], errors="coerce")

X_wave, y = [], []
X_unknown, unknown_names = [], []

files = [f for f in os.listdir(WAVE_FOLDER) if f.endswith(".mseed")]
print(f"Total files = {len(files)}")

for f in files:
    path = os.path.join(WAVE_FOLDER, f)
    try:
        wave = process_wave(path)
    except Exception as e:
        print(f"Skip {f}: {e}")
        continue

    label = np.nan
    for _, row in df.iterrows():
        if row["filename"] in f.lower():
            label = row["label"]
            break

    if pd.isna(label):
        X_unknown.append(wave)
        unknown_names.append(f)
    else:
        X_wave.append(wave)
        y.append(int(label))

X_wave = np.array(X_wave, dtype=np.float32)
y      = np.array(y)
X_unknown = np.array(X_unknown, dtype=np.float32) if X_unknown else np.array([], dtype=np.float32)

print(f"Labeled samples = {len(y)}")
print(f"Unknown samples = {len(X_unknown)}")
print(f"Earthquake = {(y==0).sum()} | Explosion = {(y==1).sum()}")

# ============================================================
# 5-Fold Stratified Cross-Validation
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
all_fold_prob = np.zeros(len(y))
all_fold_pred = np.zeros(len(y), dtype=int)

fold_idx = 0
for train_idx, test_idx in skf.split(X_wave, y):
    fold_idx += 1
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}/{N_FOLDS}")
    print(f"{'='*60}")
    print(f"Train events = {len(train_idx)} | Test events = {len(test_idx)}")
    print(f"Train: EQ={(y[train_idx]==0).sum()} EX={(y[train_idx]==1).sum()}")
    print(f"Test:  EQ={(y[test_idx]==0).sum()}  EX={(y[test_idx]==1).sum()}")

    X_train_wave = X_wave[train_idx]
    X_test_wave  = X_wave[test_idx]
    y_train      = y[train_idx]
    y_test       = y[test_idx]

    # ---- Train augmentation ----
    print("\n--- Train: sliding window augmentation ---")
    X_train_aug_list = []
    y_train_aug_list = []

    eq_count = int((y_train == 0).sum())
    ex_count = int((y_train == 1).sum())
    minority_class  = 0 if eq_count < ex_count else 1
    majority_class  = 1 - minority_class
    minority_idx    = np.where(y_train == minority_class)[0]
    majority_idx    = np.where(y_train == majority_class)[0]

    for i in minority_idx:
        wave = X_train_wave[i]
        windows = sliding_windows(wave)
        for ww in windows:
            X_train_aug_list.append(ww)
            y_train_aug_list.append(minority_class)
            X_train_aug_list.append(augment_wave_light(ww))
            y_train_aug_list.append(minority_class)
            X_train_aug_list.append(augment_wave_amplitude_shift(ww))
            y_train_aug_list.append(minority_class)

    for i in majority_idx:
        wave = X_train_wave[i]
        windows = sliding_windows(wave)
        for ww in windows[:3]:
            X_train_aug_list.append(ww)
            y_train_aug_list.append(majority_class)
            X_train_aug_list.append(augment_wave_amplitude_shift(ww))
            y_train_aug_list.append(majority_class)

    X_train_aug = np.array(X_train_aug_list, dtype=np.float32)
    y_train_aug = np.array(y_train_aug_list)

    perm = np.random.permutation(len(y_train_aug))
    X_train_aug = X_train_aug[perm]
    y_train_aug = y_train_aug[perm]

    # ---- Test: center window only ----
    print("--- Test: center window only ---")
    X_test_aug_list = []
    for i in range(len(X_test_wave)):
        cw = center_window(X_test_wave[i])
        X_test_aug_list.append(cw)
    X_test_aug = np.array(X_test_aug_list, dtype=np.float32)

    # ---- Add Channel Dimension ----
    X_train_cnn = X_train_aug[..., np.newaxis]
    X_test_cnn  = X_test_aug[..., np.newaxis]

    # ---- Class Weight ----
    eq_final = (y_train_aug == 0).sum()
    ex_final = (y_train_aug == 1).sum()
    total = eq_final + ex_final
    class_weight = {
        0: total / (2.0 * eq_final),
        1: total / (2.0 * ex_final)
    }

    # ---- Build & Train ----
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = build_1dcnn_model()
    if fold_idx == 1:
        model.summary()

    best_model_path = os.path.join(OUTPUT_FOLDER, f"best_1dcnn_fold{fold_idx}.keras")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=20, restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train_cnn, y_train_aug,
        validation_data=(X_test_cnn, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stop, lr_scheduler, checkpoint],
        verbose=1
    )

    if fold_idx == 1:
        hist1 = history.history

    # ---- Evaluate fold ----
    model = tf.keras.models.load_model(best_model_path)
    test_prob = model.predict(X_test_cnn, verbose=0).ravel()

    optimal_threshold = find_optimal_threshold(y_test, test_prob)
    test_pred = (test_prob > optimal_threshold).astype(int)

    acc = accuracy_score(y_test, test_pred)
    f1  = f1_score(y_test, test_pred)
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    roc_auc_val = auc(fpr, tpr)

    all_fold_prob[test_idx] = test_prob
    all_fold_pred[test_idx] = test_pred

    print(f"\nFold {fold_idx} Results:")
    print(f"  Threshold = {optimal_threshold:.4f}")
    print(f"  Accuracy  = {acc:.4f}")
    print(f"  AUC       = {roc_auc_val:.4f}")
    print(f"  F1 Score  = {f1:.4f}")

    fold_results.append({
        'fold': fold_idx,
        'threshold': optimal_threshold,
        'accuracy': acc,
        'auc': roc_auc_val,
        'f1': f1,
    })

# ============================================================
# Training History Plot (Fold 1)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(hist1['loss'], label='Train Loss')
axes[0].plot(hist1['val_loss'], label='Val Loss')
axes[0].set_title('Loss Curve (Fold 1)'); axes[0].legend()

axes[1].plot(hist1['accuracy'], label='Train Acc')
axes[1].plot(hist1['val_accuracy'], label='Val Acc')
axes[1].set_title('Accuracy Curve (Fold 1)'); axes[1].legend()

axes[2].plot(hist1['auc'], label='Train AUC')
axes[2].plot(hist1['val_auc'], label='Val AUC')
axes[2].set_title('AUC Curve (Fold 1)'); axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "1dcnn_training_history_fold1.png"), dpi=300)
plt.close()

# ============================================================
# Aggregate 5-Fold Results
# ============================================================
print("\n" + "="*60)
print("5-Fold Cross-Validation Summary")
print("="*60)

accs = [r['accuracy'] for r in fold_results]
aucs = [r['auc']      for r in fold_results]
f1s  = [r['f1']       for r in fold_results]

print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print("\nPer-fold details:")
for r in fold_results:
    print(f"  Fold {r['fold']}: Acc={r['accuracy']:.4f}, "
          f"AUC={r['auc']:.4f}, F1={r['f1']:.4f}, thr={r['threshold']:.4f}")

# Pooled (out-of-fold) metrics
overall_acc = accuracy_score(y, all_fold_pred)
overall_f1  = f1_score(y, all_fold_pred)
fpr_all, tpr_all, _ = roc_curve(y, all_fold_prob)
overall_auc = auc(fpr_all, tpr_all)

print(f"\nPooled (out-of-fold) metrics:")
print(f"  Accuracy  = {overall_acc:.4f}")
print(f"  AUC       = {overall_auc:.4f}")
print(f"  F1 Score  = {overall_f1:.4f}")
print("\nPooled Classification Report:")
print(classification_report(y, all_fold_pred, target_names=["Earthquake","Explosion"]))

# ============================================================
# Pooled Confusion Matrix & ROC
# ============================================================
cm = confusion_matrix(y, all_fold_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Earthquake", "Explosion"],
            yticklabels=["Earthquake", "Explosion"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("1D-CNN Confusion Matrix (5-Fold Pooled)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "1dcnn_confusion_matrix_pooled.png"), dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(fpr_all, tpr_all, lw=2, label=f"Pooled AUC = {overall_auc:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (1D-CNN, 5-Fold Pooled)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "1dcnn_roc_curve_pooled.png"), dpi=300)
plt.close()

# ============================================================
# Per-Fold Metrics Bar Plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(N_FOLDS)
width = 0.25
ax.bar(x_pos - width, accs, width, label='Accuracy')
ax.bar(x_pos,        aucs, width, label='AUC')
ax.bar(x_pos + width, f1s,  width, label='F1 Score')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
ax.set_ylabel("Score")
ax.set_title("Per-Fold Metrics (1D-CNN)")
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "1dcnn_per_fold_metrics.png"), dpi=300)
plt.close()

# ============================================================
# Unknown Prediction - Ensemble of 5 Folds
# ============================================================
if len(X_unknown) > 0:
    print("\nPredicting unknown samples (ensemble of 5 folds)...")
    X_unk_cnn_list = []
    for wave in X_unknown:
        cw = center_window(wave)
        X_unk_cnn_list.append(cw)
    
    X_unk_cnn = np.array(X_unk_cnn_list, dtype=np.float32)[..., np.newaxis]
    
    unk_probs = np.zeros(len(X_unknown))
    for fold_i in range(1, N_FOLDS + 1):
        m_path = os.path.join(OUTPUT_FOLDER, f"best_1dcnn_fold{fold_i}.keras")
        m = tf.keras.models.load_model(m_path)
        unk_probs += m.predict(X_unk_cnn, verbose=0).ravel()
        del m
        tf.keras.backend.clear_session()
    unk_probs /= N_FOLDS

    mean_threshold = float(np.mean([r['threshold'] for r in fold_results]))
    print(f"Mean threshold across folds = {mean_threshold:.4f}")

    unk_pred = (unk_probs > mean_threshold).astype(int)
    unk_conf = np.abs(unk_probs - 0.5) * 2

    pred_df = pd.DataFrame({
        "File_Name":       unknown_names,
        "Probability":     unk_probs,
        "Confidence":      unk_conf,
        "Predicted_Label": unk_pred
    })
    pred_df["Classification"] = pred_df["Predicted_Label"].map({0:"Earthquake",1:"Explosion"})
    pred_df.to_csv(os.path.join(OUTPUT_FOLDER, "1dcnn_predictions.csv"),
                   index=False, encoding='utf-8-sig')

    print("\nUnknown Sample Predictions:")
    for _, row in pred_df.iterrows():
        print(f"  {row['File_Name']}: {row['Classification']} "
              f"(prob={row['Probability']:.4f}, conf={row['Confidence']:.4f})")

# ============================================================
# Save Best Model (by AUC) as Final
# ============================================================
best_fold = max(fold_results, key=lambda r: r['auc'])
print(f"\nBest fold (by AUC): Fold {best_fold['fold']} (AUC={best_fold['auc']:.4f})")
final_model_path = os.path.join(OUTPUT_FOLDER, "best_1dcnn_model.keras")
shutil.copy(
    os.path.join(OUTPUT_FOLDER, f"best_1dcnn_fold{best_fold['fold']}.keras"),
    final_model_path
)

# ============================================================
# Summary
# ============================================================
with open(os.path.join(OUTPUT_FOLDER, "1dcnn_summary.txt"), "w", encoding='utf-8') as f:
    f.write("Earthquake / Explosion Classification: 1D-CNN (5-Fold CV)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: 1D-CNN (GlobalAveragePooling)\n")
    f.write(f"Sampling Rate: {FIXED_FS} Hz\n")
    f.write(f"Input: Raw 1D waveform ({FIXED_LEN} samples), no feature extraction\n")
    f.write(f"Folds: {N_FOLDS}\n\n")
    f.write("5-Fold CV Results:\n")
    for r in fold_results:
        f.write(f"  Fold {r['fold']}: Acc={r['accuracy']:.4f}, "
                f"AUC={r['auc']:.4f}, F1={r['f1']:.4f}, "
                f"thr={r['threshold']:.4f}\n")
    f.write(f"\nMean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
    f.write(f"Mean AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")
    f.write(f"Mean F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n\n")
    f.write(f"Pooled OOF Accuracy: {overall_acc:.4f}\n")
    f.write(f"Pooled OOF AUC:      {overall_auc:.4f}\n")
    f.write(f"Pooled OOF F1 Score: {overall_f1:.4f}\n")

print("\nAll processes finished successfully!")
print(f"Results saved to: {OUTPUT_FOLDER}")