import os
import pandas as pd
import glob
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 이름에서 레이블을 결정하는 함수
def get_label(filename):
    if 'normal' in filename:
        return 0
    else:
        return 1

# CSV 파일이 있는 디렉토리
csv_directory = '.'

# result.csv를 제외한 모든 CSV 파일 목록 생성
csv_files = glob.glob(os.path.join(csv_directory, 'anomaly_score_*.csv'))

# 실제 레이블과 anomaly score를 저장할 리스트 초기화
actual_labels = []
anomaly_scores = []

# 각 CSV 파일 읽기
for file in csv_files:
    df = pd.read_csv(file, header=None)
    df.columns = ['filename', 'anomaly_score']
    df['label'] = df['filename'].apply(get_label)
    actual_labels.extend(df['label'].tolist())
    anomaly_scores.extend(df['anomaly_score'].tolist())

# 리스트를 sklearn과 호환되는 pandas Series로 변환
actual_labels = pd.Series(actual_labels)
anomaly_scores = pd.Series(anomaly_scores)

# 임계값을 사용하여 점수를 이진화 (예: 3.0)
threshold = 3.0
predicted_labels = (anomaly_scores > threshold).astype(int)

# F1 점수 계산
f1 = f1_score(actual_labels, predicted_labels)
print(f'F1 점수: {f1}')

# 혼동 행렬 생성
cm = confusion_matrix(actual_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'anomaly'], yticklabels=['normal', 'anomaly'])
plt.xlabel('pred')
plt.ylabel('actual')

# 플롯을 파일로 저장
plt.savefig('confusion_matrix.png')

# plt.show()는 주석 처리 또는 삭제
# plt.show()
