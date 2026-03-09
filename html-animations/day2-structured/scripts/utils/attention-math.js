function dotProduct(a, b) {
  const len = Math.min(Array.isArray(a) ? a.length : 0, Array.isArray(b) ? b.length : 0);
  let sum = 0;
  for (let i = 0; i < len; i += 1) {
    sum += (Number(a[i]) || 0) * (Number(b[i]) || 0);
  }
  return sum;
}

function softmax(values) {
  if (!Array.isArray(values) || values.length === 0) return [];
  const maxVal = Math.max.apply(null, values);
  const exps = values.map((value) => Math.exp(value - maxVal));
  const denom = exps.reduce((acc, value) => acc + value, 0);
  if (denom <= 0) return values.map(() => 0);
  return exps.map((value) => value / denom);
}

function normalizeNegativeZero(value) {
  return Object.is(value, -0) ? 0 : value;
}

function formatVectorValue(value) {
  return normalizeNegativeZero(Number(value) || 0).toFixed(1);
}

function formatScoreValue(value) {
  return normalizeNegativeZero(Number(value) || 0).toFixed(1);
}

function formatWeightValue(value) {
  return normalizeNegativeZero(Number(value) || 0).toFixed(2);
}

function formatAttentionMatrixScoreCellValue(value, mode = 'score') {
  if (mode === 'masked' && !Number.isFinite(value)) return '\u2212\u221e';
  if (mode === 'probability') return formatWeightValue(value);
  return formatScoreValue(value);
}

function computeAttentionMatrixRawScores() {
  return Object.fromEntries(
    ATTN_MATRIX_TOKENS.map((rowToken) => [
      rowToken,
      Object.fromEntries(
        ATTN_MATRIX_SCORE_TOKENS.map((colToken) => [
          colToken,
          dotProduct(
            ATTN_MATRIX_Q_VECTORS[rowToken] || [],
            ATTN_QKV_KEY_VECTORS[colToken] || []
          )
        ])
      )
    ])
  );
}

function computeAttentionMatrixScaledScores() {
  const scale = Math.sqrt(ATTN_MATRIX_D_K) || 1;
  return Object.fromEntries(
    ATTN_MATRIX_TOKENS.map((rowToken) => [
      rowToken,
      Object.fromEntries(
        ATTN_MATRIX_SCORE_TOKENS.map((colToken) => [
          colToken,
          (ATTN_MATRIX_SCORE_RAW_ROWS[rowToken] || {})[colToken] / scale
        ])
      )
    ])
  );
}

function computeAttentionMatrixRawMaskedScores() {
  return Object.fromEntries(
    ATTN_MATRIX_TOKENS.map((rowToken) => [
      rowToken,
      Object.fromEntries(
        ATTN_MATRIX_SCORE_TOKENS.map((colToken) => [
          colToken,
          (ATTN_MATRIX_CAUSAL_MASK[rowToken] || {})[colToken]
            ? Number.NEGATIVE_INFINITY
            : ((ATTN_MATRIX_SCORE_RAW_ROWS[rowToken] || {})[colToken])
        ])
      )
    ])
  );
}

function computeAttentionMatrixCausalMask() {
  return Object.fromEntries(
    ATTN_MATRIX_TOKENS.map((rowToken, rowIdx) => [
      rowToken,
      Object.fromEntries(
        ATTN_MATRIX_SCORE_TOKENS.map((colToken, colIdx) => [colToken, colIdx > rowIdx])
      )
    ])
  );
}

function computeAttentionMatrixMaskedScores() {
  return Object.fromEntries(
    ATTN_MATRIX_TOKENS.map((rowToken) => [
      rowToken,
      Object.fromEntries(
        ATTN_MATRIX_SCORE_TOKENS.map((colToken) => [
          colToken,
          (ATTN_MATRIX_CAUSAL_MASK[rowToken] || {})[colToken]
            ? Number.NEGATIVE_INFINITY
            : ((ATTN_MATRIX_SCORE_SCALED_ROWS[rowToken] || {})[colToken])
        ])
      )
    ])
  );
}

function computeAttentionMatrixWeights() {
  return Object.fromEntries(
    ATTN_MATRIX_TOKENS.map((rowToken) => {
      const maskedRow = ATTN_MATRIX_SCORE_TOKENS.map(
        (colToken) => ((ATTN_MATRIX_SCORE_MASKED_ROWS[rowToken] || {})[colToken])
      );
      const weights = softmax(maskedRow);
      return [
        rowToken,
        Object.fromEntries(
          ATTN_MATRIX_SCORE_TOKENS.map((colToken, idx) => [colToken, weights[idx] || 0])
        )
      ];
    })
  );
}

function computeAttentionMatrixOutputs() {
  return Object.fromEntries(
    ATTN_MATRIX_TOKENS.map((rowToken) => {
      const outVector = Array.from({ length: ATTN_QKV_QUERY_VECTOR.length }, (_, dimIdx) => (
        ATTN_MATRIX_SCORE_TOKENS.reduce((sum, colToken) => (
          sum + (((ATTN_MATRIX_ATTN_ROWS[rowToken] || {})[colToken] || 0)
            * (((ATTN_QKV_VALUE_VECTORS[colToken] || [])[dimIdx]) || 0))
        ), 0)
      ));
      return [rowToken, outVector];
    })
  );
}

function sliceAttentionMultiHeadRows(sourceRows, dims) {
  return Object.fromEntries(
    ATTN_QKV_TOKENS.map((token) => [
      token,
      (Array.isArray(dims) ? dims : []).map((dimIdx) => (((sourceRows[token] || [])[dimIdx]) || 0))
    ])
  );
}

function computeAttentionMultiHeadScores(qRows, kRows) {
  return Object.fromEntries(
    ATTN_QKV_TOKENS.map((rowToken) => [
      rowToken,
      Object.fromEntries(
        ATTN_QKV_TOKENS.map((colToken) => [
          colToken,
          dotProduct(qRows[rowToken] || [], kRows[colToken] || [])
        ])
      )
    ])
  );
}

function computeAttentionMultiHeadMaskedScores(scoreRows) {
  const scale = Math.sqrt(ATTN_MHA_HEAD_DIM) || 1;
  return Object.fromEntries(
    ATTN_QKV_TOKENS.map((rowToken, rowIdx) => [
      rowToken,
      Object.fromEntries(
        ATTN_QKV_TOKENS.map((colToken, colIdx) => [
          colToken,
          colIdx > rowIdx
            ? Number.NEGATIVE_INFINITY
            : (((scoreRows[rowToken] || {})[colToken] || 0) / scale)
        ])
      )
    ])
  );
}

function computeAttentionMultiHeadWeights(maskedRows) {
  return Object.fromEntries(
    ATTN_QKV_TOKENS.map((rowToken) => {
      const maskedValues = ATTN_QKV_TOKENS.map((colToken) => ((maskedRows[rowToken] || {})[colToken]));
      const weights = softmax(maskedValues);
      return [
        rowToken,
        Object.fromEntries(
          ATTN_QKV_TOKENS.map((colToken, idx) => [colToken, weights[idx] || 0])
        )
      ];
    })
  );
}

function computeAttentionMultiHeadOutputs(attnRows, vRows) {
  const outDim = ((ATTN_MHA_HEAD_SLICES[ATTN_MHA_HEADS[0]] || []).length) || 0;
  return Object.fromEntries(
    ATTN_QKV_TOKENS.map((rowToken) => [
      rowToken,
      Array.from({ length: outDim }, (_, dimIdx) => (
        ATTN_QKV_TOKENS.reduce((sum, colToken) => (
          sum
          + ((((attnRows[rowToken] || {})[colToken]) || 0) * ((((vRows[colToken] || [])[dimIdx]) || 0)))
        ), 0)
      ))
    ])
  );
}

function initializeAttentionDerivedConstants() {
  ATTN_MATRIX_SCORE_RAW_ROWS = computeAttentionMatrixRawScores();
  ATTN_MATRIX_SCORE_SCALED_ROWS = computeAttentionMatrixScaledScores();
  ATTN_MATRIX_CAUSAL_MASK = computeAttentionMatrixCausalMask();
  ATTN_MATRIX_SCORE_RAW_MASKED_ROWS = computeAttentionMatrixRawMaskedScores();
  ATTN_MATRIX_SCORE_MASKED_ROWS = computeAttentionMatrixMaskedScores();
  ATTN_MATRIX_ATTN_ROWS = computeAttentionMatrixWeights();
  ATTN_MATRIX_OUTPUT_ROWS = computeAttentionMatrixOutputs();

  ATTN_MHA_Q_ROWS = Object.fromEntries(
    ATTN_MHA_HEADS.map((head) => [head, sliceAttentionMultiHeadRows(ATTN_MATRIX_Q_VECTORS, ATTN_MHA_HEAD_SLICES[head])])
  );
  ATTN_MHA_K_ROWS = Object.fromEntries(
    ATTN_MHA_HEADS.map((head) => [head, sliceAttentionMultiHeadRows(ATTN_QKV_KEY_VECTORS, ATTN_MHA_HEAD_SLICES[head])])
  );
  ATTN_MHA_V_ROWS = Object.fromEntries(
    ATTN_MHA_HEADS.map((head) => [head, sliceAttentionMultiHeadRows(ATTN_QKV_VALUE_VECTORS, ATTN_MHA_HEAD_SLICES[head])])
  );
  ATTN_MHA_SCORE_ROWS = Object.fromEntries(
    ATTN_MHA_HEADS.map((head) => [head, computeAttentionMultiHeadScores(ATTN_MHA_Q_ROWS[head], ATTN_MHA_K_ROWS[head])])
  );
  ATTN_MHA_MASKED_ROWS = Object.fromEntries(
    ATTN_MHA_HEADS.map((head) => [head, computeAttentionMultiHeadMaskedScores(ATTN_MHA_SCORE_ROWS[head])])
  );
  ATTN_MHA_ATTN_ROWS = Object.fromEntries(
    ATTN_MHA_HEADS.map((head) => [head, computeAttentionMultiHeadWeights(ATTN_MHA_MASKED_ROWS[head])])
  );
  ATTN_MHA_OUTPUT_ROWS = Object.fromEntries(
    ATTN_MHA_HEADS.map((head) => [head, computeAttentionMultiHeadOutputs(ATTN_MHA_ATTN_ROWS[head], ATTN_MHA_V_ROWS[head])])
  );

  ATTN_WGT_RAW_SCORE_BY_TOKEN = Object.fromEntries(
    ATTN_WGT_TOKENS.map((token) => [
      token,
      dotProduct(ATTN_QKV_QUERY_VECTOR, ATTN_QKV_KEY_VECTORS[token] || [])
    ])
  );
  ATTN_WGT_RAW_SCORES = ATTN_WGT_TOKENS.map((token) => ATTN_WGT_RAW_SCORE_BY_TOKEN[token]);
  ATTN_WGT_SCALED_SCORES = ATTN_WGT_RAW_SCORES.map((score) => score / Math.sqrt(ATTN_WGT_D_K));
  ATTN_WGT_WEIGHTS = softmax(ATTN_WGT_SCALED_SCORES);

  ATTN_STEP4_SCORE_BY_TOKEN = Object.fromEntries(
    ATTN_STEP4_TOKENS.map((token) => [token, ATTN_WGT_RAW_SCORE_BY_TOKEN[token] || 0])
  );
  ATTN_STEP4_WEIGHT_BY_TOKEN = Object.fromEntries(
    ATTN_STEP4_TOKENS.map((token) => {
      const idx = ATTN_WGT_TOKENS.indexOf(token);
      return [token, idx >= 0 ? ATTN_WGT_WEIGHTS[idx] : 0];
    })
  );
  ATTN_STEP4_WEIGHTED_VALUE_VECTORS = Object.fromEntries(
    ATTN_STEP4_TOKENS.map((token) => {
      const weight = ATTN_STEP4_WEIGHT_BY_TOKEN[token] || 0;
      const baseValues = ATTN_QKV_VALUE_VECTORS[token] || [];
      return [token, baseValues.map((value) => value * weight)];
    })
  );
  ATTN_STEP4_AGG_VECTOR = ATTN_QKV_QUERY_VECTOR.map((_, idx) => (
    ATTN_STEP4_TOKENS.reduce(
      (sum, token) => sum + (((ATTN_STEP4_WEIGHTED_VALUE_VECTORS[token] || [])[idx]) || 0),
      0
    )
  ));
  ATTN_STEP4_RESIDUAL_INPUT_VECTOR = (ATTN_QKV_X_VECTORS[ATTN_STEP4_FOCUS] || []).slice();
  ATTN_STEP4_RESIDUAL_OUTPUT_VECTOR = ATTN_STEP4_AGG_VECTOR.map(
    (value, idx) => (Number(ATTN_STEP4_RESIDUAL_INPUT_VECTOR[idx]) || 0) + (Number(value) || 0)
  );
}

initializeAttentionDerivedConstants();
