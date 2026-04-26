import json
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

print("1. Загружаем логи...")
logs = []
with open('botify/log/data.json', 'r') as f:
    for line in f:
        try:
            logs.append(json.loads(line))
        except:
            continue

df = pd.DataFrame(logs)
print(f"Загружено {len(df)} событий")

# Загружаем метаданные треков
tracks_meta = {}
with open('sim/data/tracks.json', 'r') as f:
    for line in f:
        try:
            track = json.loads(line)
            tracks_meta[track['track']] = track
        except:
            continue

print(f"Загружено треков: {len(tracks_meta)}")

print("\n2. Загружаем эмбеддинги...")
embeddings = np.load('sim/data/embeddings.npy')
embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
print(f"Эмбеддинги: {embeddings.shape}")

print("\n3. Собираем обучающие примеры проще и надёжнее...")

# Собираем историю каждого пользователя
user_history = defaultdict(list)
for _, row in df.iterrows():
    user = int(row['user'])
    user_history[user].append({
        'track': int(row['track']),
        'time': float(row['time']),
        'recommendation': int(row['recommendation']) if pd.notna(row.get('recommendation')) and row['recommendation'] is not None else None
    })

training_data = []
max_track_id = len(embeddings)

for user, history in user_history.items():
    if len(history) < 5:
        continue
    
    # User-level статистика
    user_times = [h['time'] for h in history]
    user_avg_time = np.mean(user_times)
    user_std_time = np.std(user_times) if len(user_times) > 1 else 0.1
    
    for i in range(5, len(history)):
        # История до текущего момента
        past = history[max(0, i-15):i]
        current = history[i]
        
        current_track = current['track']
        current_time = current['time']
        recommended_track = current['recommendation']
        
        # Проверки валидности
        if recommended_track is None:
            continue
        if current_track >= max_track_id or recommended_track >= max_track_id:
            continue
        
        past_tracks = [h['track'] for h in past if h['track'] < max_track_id]
        past_times = [h['time'] for h in past]
        
        if len(past_tracks) == 0:
            continue
        
        # Target
        liked = 1 if current_time > 0.5 else 0
        
        # === ПРИЗНАКИ ===
        
        # 1. Средний эмбеддинг истории с временными весами
        n_past = len(past_tracks)
        time_weights = np.exp(-0.1 * np.arange(n_past)[::-1])
        time_weights = time_weights[:n_past]
        time_weights = time_weights / (time_weights.sum() + 1e-8)
        
        history_emb = np.sum(embeddings_norm[past_tracks] * time_weights.reshape(-1, 1), axis=0)
        
        # Эмбеддинги
        track_emb = embeddings_norm[current_track]
        rec_emb = embeddings_norm[recommended_track]
        
        cos_hist_current = np.dot(history_emb, track_emb)
        cos_hist_rec = np.dot(history_emb, rec_emb)
        cos_current_rec = np.dot(track_emb, rec_emb)
        
        features = {
            # Эмбеддинг-фичи
            'cos_hist_current': cos_hist_current,
            'cos_hist_rec': cos_hist_rec,
            'cos_current_rec': cos_current_rec,
            'hist_emb_norm': np.linalg.norm(history_emb),
            
            # Сессионные
            'history_length': n_past,
            'current_time': current_time,
            'session_position': i / len(history),
            'avg_history_time': np.mean(past_times[-10:]) if len(past_times) >= 1 else 0.5,
            'std_history_time': np.std(past_times[-10:]) if len(past_times) >= 2 else 0,
            'skip_ratio': sum(1 for t in past_times[-10:] if t < 0.3) / min(10, n_past),
            
            # User-level
            'user_avg_time': user_avg_time,
            'user_std_time': user_std_time,
            'user_history_len': len(history),
            
            # Разнообразие
            'unique_tracks_ratio': len(set(past_tracks[-10:])) / min(10, n_past),
            'avg_pairwise_sim': np.mean([np.dot(embeddings_norm[past_tracks[j]], embeddings_norm[past_tracks[k]]) 
                                          for j in range(min(10, n_past)) 
                                          for k in range(j+1, min(10, n_past))]) if n_past >= 2 else 0,
        }
        
        # Топ-5 компонент эмбеддинга
        for j in range(min(8, len(history_emb))):
            features[f'hist_emb_{j}'] = float(history_emb[j])
        
        # Метаданные рекомендованного трека
        if recommended_track in tracks_meta:
            rec_meta = tracks_meta[recommended_track]
            features['rec_year'] = rec_meta.get('year', 2000)
            features['rec_fans'] = rec_meta.get('artist_fans', 50)
        
        training_data.append({**features, 'liked': liked})

df_train = pd.DataFrame(training_data)
print(f"\nСоздано {len(df_train)} обучающих примеров")
print(f"Признаков: {len(df_train.columns) - 1}")

if len(df_train) == 0:
    print("ОШИБКА: нет данных для обучения!")
    exit()

print(f"Баланс классов:")
print(df_train['liked'].value_counts(normalize=True))

print("\n4. Обучаем LightGBM...")

if 'current_time' in df_train.columns:
    df_train = df_train.drop('current_time', axis=1)
    
# Исправляем типы данных
# Заполняем пропуски
df_train = df_train.fillna(0)

# Приводим все колонки к numeric
for col in df_train.columns:
    if col != 'liked':
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)

X = df_train.drop('liked', axis=1)
y = df_train['liked'].astype(int)

# Удаляем константные признаки
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"Удаляем константные признаки: {constant_cols}")
    X = X.drop(constant_cols, axis=1)

# Убеждаемся что все float
X = X.astype(float)

print(f"Финальный размер: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(30, verbose=10)]
)

# Оценка
y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"\n=== Результаты ===")
print(f"ROC-AUC: {auc:.4f}")

print(f"\nTop-10 важных признаков:")
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print(importance.to_string())

# Сохраняем
print("\n5. Сохраняем...")
import os
os.makedirs('botify/botify', exist_ok=True)

with open('botify/botify/model.pkl', 'wb') as f:
    pickle.dump(model, f)

np.save('botify/botify/embeddings.npy', embeddings_norm)

with open('botify/botify/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

with open('botify/botify/tracks_meta.pkl', 'wb') as f:
    pickle.dump(tracks_meta, f)

print("\n✅ Готово! Файлы в botify/botify/")