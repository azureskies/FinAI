/**
 * Technical indicators dictionary with Chinese descriptions and interpretations.
 */

export interface IndicatorInfo {
	name: string;
	description: string;
	interpretation: string;
}

export const INDICATOR_DICT: Record<string, IndicatorInfo> = {
	adj_close: {
		name: '還原收盤價',
		description: '考慮了除權息後的價格。',
		interpretation: '反映真實持股成本與獲利，用於精確計算回測。'
	},
	sma_20: {
		name: '20日均線 (月線)',
		description: '過去 20 個交易日的平均收盤價。',
		interpretation: '短線支撐參考。股價在均線上為強勢，在下為弱勢。'
	},
	sma_60: {
		name: '60日均線 (季線)',
		description: '過去 60 個交易日的平均收盤價。',
		interpretation: '中期多空分界線。季線向上通常代表趨勢偏多。'
	},
	sma_120: {
		name: '120日均線 (半年線)',
		description: '過去 120 個交易日的平均收盤價。',
		interpretation: '長期趨勢參考。'
	},
	ema_12: {
		name: '12日指數均線',
		description: '對近期價格權重較高的均線。',
		interpretation: '反應比 SMA 靈敏，用於觀察短期動能轉折。'
	},
	ema_26: {
		name: '26日指數均線',
		description: 'MACD 計算中的長期基準。',
		interpretation: '與短天期 EMA 配合觀察黃金/死亡交叉。'
	},
	ema_50: {
		name: '50日指數均線',
		description: '中期趨勢基準。',
		interpretation: '常用於過濾短線噪音，確認趨勢方向。'
	},
	adx_14: {
		name: '平均方向指數',
		description: '衡量趨勢強度（不論漲跌）。',
		interpretation: '高於 25 代表趨勢強烈；低於 20 代表盤整。'
	},
	rsi_14: {
		name: '14日相對強弱指標',
		description: '衡量漲跌力道對比。',
		interpretation: '70 以上超買，30 以下超賣。'
	},
	rsi_28: {
		name: '28日相對強弱指標',
		description: '長天期強弱指標。',
		interpretation: '反應較慢，適合觀察中長線超買超賣。'
	},
	macd_line: {
		name: 'MACD 快線',
		description: 'EMA(12) 與 EMA(26) 的差值。',
		interpretation: '向上突破訊號線視為買點。'
	},
	macd_signal: {
		name: 'MACD 訊號線 (慢線)',
		description: 'MACD 快線的 9 日平均。',
		interpretation: '輔助快線確認趨勢轉向。'
	},
	macd_histogram: {
		name: 'MACD 柱狀圖',
		description: '快線與慢線的差值。',
		interpretation: '由負轉正代表動能增強；柱狀圖拉長代表力道加劇。'
	},
	stoch_k: {
		name: 'KD-K值',
		description: '隨機指標 K 值，反應當前位置。',
		interpretation: '通常 K > D 代表短線偏多。'
	},
	stoch_d: {
		name: 'KD-D值',
		description: 'K 值的平滑移動平均。',
		interpretation: '輔助判斷超買超賣區。'
	},
	stoch_rsi_k: {
		name: 'Stochastic RSI K值',
		description: '對 RSI 再進行一次 KD 計算。',
		interpretation: '反應極其靈敏，適合捕捉盤整區間的極短線轉折。'
	},
	stoch_rsi_d: {
		name: 'Stochastic RSI D值',
		description: 'Stoch RSI 的訊號線。',
		interpretation: '與 K 值配合抓取短線轉折點。'
	},
	roc_5: {
		name: '5日價格變動率',
		description: '目前價格與 5 天前價格的變化百分比。',
		interpretation: '大於 0 代表短線動能向上。'
	},
	roc_10: {
		name: '10日價格變動率',
		description: '衡量中短期爆發力。',
		interpretation: '上升趨勢中 ROC 快速拉升代表多頭攻擊強勁。'
	},
	roc_20: {
		name: '20日價格變動率',
		description: '衡量月度級別的價格變化。',
		interpretation: '用於確認中期趨勢動能。'
	},
	momentum_5: {
		name: '5日動能指標',
		description: '比較當前與 5 天前的價格差。',
		interpretation: '反映短線買賣盤力道。'
	},
	momentum_10: {
		name: '10日動能指標',
		description: '反映中短線動能。',
		interpretation: '趨勢向上時動能數值應持續為正。'
	},
	momentum_20: {
		name: '20日動能指標',
		description: '反映月中期動能。',
		interpretation: '用於過濾短線雜訊的動能指標。'
	},
	bb_upper: {
		name: '布林上軌',
		description: '20MA 加上兩個標準差。',
		interpretation: '股價觸碰上軌通常視為超買或壓力區。'
	},
	bb_lower: {
		name: '布林下軌',
		description: '20MA 減去兩個標準差。',
		interpretation: '股價觸碰下軌通常視為超賣或支撐區。'
	},
	bb_width: {
		name: '布林帶寬度',
		description: '衡量股價波動區間的大小。',
		interpretation: '寬度極小時常預示即將發生大幅變盤。'
	},
	atr_14: {
		name: '14日平均真實波幅',
		description: '衡量股價的平均波動幅度。',
		interpretation: '數值越高波动越大，可用於設定停損範圍。'
	},
	atr_28: {
		name: '28日平均真實波幅',
		description: '衡量較長期的波動。',
		interpretation: '幫助識別波動率的長期變化趨勢。'
	},
	hist_volatility_20: {
		name: '20日歷史波動率',
		description: '價格回報率的標準差。',
		interpretation: '越高代表風險越大，盤勢較不穩。'
	},
	obv: {
		name: '能量潮',
		description: '量價累積指標。',
		interpretation: 'OBV 與股價背離時（如價漲量縮）常預示轉折。'
	},
	cmf: {
		name: '蔡金貨幣流量指標',
		description: '衡量資金流入與流出的強度。',
		interpretation: 'CMF > 0 代表資金流入，買盤較強。'
	},
	volume_change: {
		name: '成交量變化率',
		description: '目前成交量相對於均量的倍數。',
		interpretation: '大幅增長代表有大資金進入或情緒發生劇烈變化。'
	},
	volume_sma_20: {
		name: '20日均量',
		description: '過去 20 天的平均成交量。',
		interpretation: '判斷當前成交量是否異常放大的基準。'
	},
	pct_from_52w_high: {
		name: '距52週最高點距離',
		description: '目前價格與一年最高點的差距。',
		interpretation: '接近 0 代表股價極其強勢，即將創高或已在高位。'
	},
	pct_from_52w_low: {
		name: '距52週最低點距離',
		description: '目前價格與一年最低點的漲幅。',
		interpretation: '數值越高代表從底部翻轉的幅度越大。'
	}
};

export function getIndicator(key: string): IndicatorInfo {
	return (
		INDICATOR_DICT[key] || {
			name: key,
			description: '暫無詳細說明',
			interpretation: '請參考相關技術文件。'
		}
	);
}
