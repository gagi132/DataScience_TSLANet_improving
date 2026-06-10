"""
분할 기준
    Train : 70%  → 676 rows
    Val   : 10%  →  97 rows
    Test  : 20%  → 193 rows
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from timefeatures import time_features

warnings.filterwarnings('ignore')


class Dataset_ILI(Dataset):
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='M',
        data_path='national_illness.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='w',
        seasonal_patterns=None,   # 호환성을 위해 유지 (미사용)
    ):
        if size is None:
            # ILI 기본값: 약 2년 입력 → 6개월 예측
            self.seq_len   = 104
            self.label_len = 18
            self.pred_len  = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type  = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features  = features
        self.target    = target
        self.scale     = scale
        self.timeenc   = timeenc
        self.freq      = freq
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    # ─────────────────────────────────────────────────────────────────────────

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ── 열 정리 ───────────────────────────────────────────────────────────
        date_col = 'date' if 'date' in df_raw.columns else df_raw.columns[0]
        df_raw = df_raw.rename(columns={date_col: 'date'})

        # OT 열 확인 (일부 버전은 '% WEIGHTED ILI' 가 OT 역할)
        if self.target not in df_raw.columns:
            ili_cols = [c for c in df_raw.columns
                        if 'WEIGHTED' in c.upper() or 'ILI' in c.upper()]
            if ili_cols:
                df_raw = df_raw.rename(columns={ili_cols[0]: 'OT'})
                self.target = 'OT'

        # 열 순서 정리: [date, ...(others), target]
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # ── Train/Val/Test 분할 (70 / 10 / 20) ───────────────────────────────
        n    = len(df_raw)
        n_tr = int(676)
        n_te = int(97)
        n_va = int(193)

        border1s = [0,         n_tr - self.seq_len, n - n_te - self.seq_len]
        border2s = [n_tr,      n_tr + n_va,         n                      ]
        b1 = border1s[self.set_type]
        b2 = border2s[self.set_type]

        # ── 특징 열 선택 ──────────────────────────────────────────────────────
        if self.features in ('M', 'MS'):
            df_data = df_raw[df_raw.columns[1:]]    # date 제외 전체
        else:
            df_data = df_raw[[self.target]]

        # ── 정규화 (train 기준 fit) ───────────────────────────────────────────
        if self.scale:
            self.scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ── 시간 특징 ─────────────────────────────────────────────────────────
        df_stamp = df_raw[['date']].iloc[b1:b2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            df_stamp['month']   = df_stamp['date'].dt.month
            df_stamp['day']     = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            # isocalendar().week 는 슬라이스 시 index 불일치 가능
            # → .to_numpy() 로 순수 배열 변환 후 int 캐스팅
            df_stamp['week'] = (
                df_stamp['date'].dt.isocalendar().week
                .to_numpy().astype(int)
            )
            data_stamp = df_stamp.drop(['date'], axis=1).astype(np.float32).values
        else:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)

        self.data_x     = data[b1:b2]
        self.data_y     = data[b1:b2]
        self.data_stamp = data_stamp

        # ── 분할 정보 출력 ────────────────────────────────────────────────────
        flag_name = ['train', 'val', 'test'][self.set_type]
        print(f"  [ILI] {flag_name}: rows {b1}~{b2} "
              f"({b2 - b1} rows, {len(self)} samples) "
              f"features={data.shape[1]}")

    # ─────────────────────────────────────────────────────────────────────────

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]
        seq_y      = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
