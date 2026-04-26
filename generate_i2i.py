import json
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("1. Загружаем модель и данные...")

# Загружаем модель
with open('botify/botify/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Загружаем эмбеддинги
embeddings = np.load('botify/botify/embeddings.npy')

# Загружаем имена признаков
with open('botify/botify/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Загружаем оригинальные SasRec I2I рекомендации
sasrec_i2i = {}
with open('botify/data/sasrec_i2i.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        sasrec_i2i[int(data['item_id'])] = [int(r) for r in data['recommendations']]

print(f"Загружено {len(sasrec_i2i)} I2I словарей")
print(f"Эмбеддинги: {embeddings.shape}")

print("\n2. Генерируем улучшенные рекомендации...")

# Для каждого трека ранжируем его I2I кандидатов моделью
improved_i2i = {}

for item_id in tqdm(sasrec_i2i.keys()):
    if item_id >= len(embeddings):
        continue
    
    # Берём кандидатов из SasRec I2I
    candidates = sasrec_i2i.get(item_id, [])
    candidates = [c for c in candidates if c < len(embeddings)]
    
    if len(candidates) <= 1:
        improved_i2i[item_id] = candidates[:10]
        continue
    
    # Для каждого кандидата извлекаем признаки
    track_emb = embeddings[item_id]
    
    features_list = []
    valid_candidates = []
    
    for cand_id in candidates:
        cand_emb = embeddings[cand_id]
        
        # Признаки (как при обучении)
        cos_hist_current = float(np.dot(track_emb, cand_emb))
        
        feat = {}
        for name in feature_names:
            if name == 'cos_hist_current':
                feat[name] = cos_hist_current
            elif name == 'cos_current_rec':
                feat[name] = cos_hist_current  # Используем item_id как историю
            elif name == 'cos_hist_rec':
                feat[name] = cos_hist_current
            elif name.startswith('hist_emb_'):
                feat[name] = 0.0
            else:
                feat[name] = 0.0
        
        features_list.append([feat.get(name, 0.0) for name in feature_names])
        valid_candidates.append(cand_id)
    
    if not valid_candidates:
        continue
    
    # Предсказываем скоры
    X = np.array(features_list, dtype=float)
    scores = model.predict_proba(X)[:, 1]
    
    # Сортируем кандидатов по скору
    sorted_candidates = [c for _, c in sorted(zip(scores, valid_candidates), key=lambda x: x[0], reverse=True)]
    
    improved_i2i[item_id] = sorted_candidates[:10]

print(f"\nСгенерировано рекомендаций для {len(improved_i2i)} треков")

# Сохраняем в JSON Lines
print("\n3. Сохраняем результат...")
with open('botify/data/improved_i2i.jsonl', 'w') as f:
    for item_id, recs in improved_i2i.items():
        f.write(json.dumps({'item_id': item_id, 'recommendations': recs}) + '\n')

print("✅ Готово! Файл: botify/data/improved_i2i.jsonl")
print(f"Пример (item_id=0): {improved_i2i.get(0, 'нет')[:5]}")