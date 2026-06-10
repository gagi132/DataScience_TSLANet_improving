import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class Dataset_METRLA(Dataset):
    def __init__(self, data_path, flag='train', seq_len=96, pred_len=96):
        """
        data_path: METR-LA.h5 파일의 절대/상대 경로
        flag: 'train', 'val', 'test' 중 하나
        seq_len: 과거 입력 길이 (예: 96 = 과거 8시간)
        pred_len: 미래 예측 길이 (예: 96 = 미래 8시간)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag
        
        # 1. 분할 비율 설정 (Train 70%, Val 10%, Test 20% - 교통량 벤치마크 표준)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.__read_data__(data_path)

    def __read_data__(self, data_path):
        # 2. HDF5 파일 로드 (Pandas 활용)
        print(f"[{self.flag}] METR-LA 데이터를 불러오는 중...")
        df_raw = pd.read_hdf(data_path)
        
        # 데이터 형태: (34272, 207) -> 34272개의 타임스텝(5분 단위), 207개의 센서
        data = df_raw.values
        
        # 3. 데이터 분할 지점 계산
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 4. 정규화 (StandardScaler)
        # 훈련 데이터 기준으로 스케일러 피팅 후 검증/테스트에 적용 (Data Leakage 방지)
        self.scaler = StandardScaler()
        train_data = data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(data)
        
        # 모델이 실제로 사용할 범위의 데이터만 잘라내기
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        # 슬라이딩 윈도우 방식으로 과거(seq_len)와 미래(pred_len) 텐서 추출
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32), 0, 0

    def __len__(self):
        # 추출 가능한 총 샘플 수
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)