const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const DEV = false;

const LENS_LABELS = {
  pet: 'Pet / companionship',
  feline: 'Feline family',
  care: 'Care context'
};

const DEFAULT_PROJECTION_LENS = 'pet';

const LENS_MEANINGS = {
  pet: 'pet and companionship features',
  feline: 'feline-family features',
  care: 'care and ownership features'
};

const LENS_STATES = {
  pet: {
    cat:[0.30,0.55],
    dog:[0.22,0.60],
    hamster:[0.20,0.45],
    vet:[0.35,0.70],
    litter:[0.48,0.30],
    kibble:[0.50,0.45],
    tiger:[0.80,0.60],
    lion:[0.82,0.45],
    leopard:[0.75,0.50]
  },
  feline: {
    cat:[0.55,0.55],
    tiger:[0.70,0.65],
    lion:[0.72,0.45],
    leopard:[0.68,0.55],
    dog:[0.20,0.65],
    hamster:[0.18,0.40],
    vet:[0.40,0.30],
    litter:[0.48,0.12],
    kibble:[0.55,0.20]
  },
  care: {
    cat:[0.45,0.55],
    litter:[0.35,0.45],
    kibble:[0.52,0.45],
    vet:[0.48,0.70],
    dog:[0.30,0.65],
    hamster:[0.32,0.30],
    tiger:[0.80,0.70],
    lion:[0.82,0.45],
    leopard:[0.75,0.55]
  }
};

const LENS_NEIGHBORS = {
  pet: ['dog', 'hamster', 'vet'],
  feline: ['leopard', 'tiger', 'lion'],
  care: ['litter', 'kibble', 'vet']
};

const LENS_EMPHASIS = {
  pet: ['cat', 'dog', 'hamster', 'vet'],
  feline: ['cat', 'tiger', 'lion', 'leopard'],
  care: ['cat', 'litter', 'kibble', 'vet']
};

const TOKEN_GROUPS = {
  cat: 'anchor',
  dog: 'pets',
  hamster: 'pets',
  tiger: 'bigcats',
  lion: 'bigcats',
  leopard: 'bigcats',
  litter: 'care',
  kibble: 'care',
  vet: 'care'
};

const TOKEN_LABEL_OFFSETS = {
  dog: [9, -12],
  hamster: [9, 12],
  tiger: [9, -12],
  lion: [9, 12],
  leopard: [9, -14],
  litter: [9, 12],
  kibble: [9, 12],
  vet: [9, -12]
};

const PROJECTION_TOKENS = Object.keys(LENS_STATES[DEFAULT_PROJECTION_LENS]);

const ATTN_INTRO_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_INTRO_FLOW_SOURCES = ['cat', 'on', 'the', 'mat', 'sat'];

const ATTN_INTRO_FOCUS = 'sat';

const ATTN_INTRO_FOOTER = 'Next: how does sat decide which tokens matter — and what to copy from them?';

const ATTN_INTRO_FLOW_HEIGHTS = {
  cat: 0.08,
  on: 0.12,
  the: 0.16,
  mat: 0.2,
  sat: 0.18
};

const ATTN_INTRO_FLOW_ANIM_MS = 380;

const ATTN_INTRO_FLOW_HEAD_FADE_MS = 150;

const ATTN_INTRO_MAX_STEP = 8;

const ATTN_QKV_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_QKV_FOCUS = 'sat';

const ATTN_QKV_QUERY_VECTOR = [1.0, 1.0, 1.0, 1.0];

const ATTN_QKV_KEY_VECTORS = {
  cat: [1.1, 0.9, 0.6, 0.4],
  sat: [0.7, 0.4, 0.3, 0.2],
  on: [0.6, 0.4, 0.2, 0.1],
  the: [0.3, 0.1, -0.1, -0.1],
  mat: [0.1, -0.1, -0.2, -0.2]
};

const ATTN_QKV_VALUE_VECTORS = {
  cat: [1.0, 0.0, 0.0, 0.0],
  sat: [1.0, 1.0, 0.0, 0.0],
  on: [0.0, 1.0, 1.0, 0.0],
  the: [0.0, 0.0, 1.0, 1.0],
  mat: [0.0, 0.0, 0.0, 1.0]
};

const ATTN_QKV_X_VECTORS = {
  cat: [0.6, 0.2, -0.1, 0.3],
  sat: [0.5, 0.7, 0.1, 0.4],
  on: [0.3, 0.6, 0.4, 0.2],
  the: [0.1, 0.4, 0.5, 0.2],
  mat: [0.0, 0.2, 0.6, 0.3]
};

const ATTN_QKV_COMPARE_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_QKV_SCORE_QUAL = {
  cat: { tier: 'strongest' },
  sat: { tier: 'medium' },
  on: { tier: 'medium' },
  the: { tier: 'weak' },
  mat: { tier: 'weak' }
};

const ATTN_QKV_TAKEAWAYS = [
  'We start from embeddings \\(x_1, \\ldots, x_n\\).',
  'Project each token embedding into a key vector: \\(k_j = x_j W_K\\).',
  'Project each token embedding into a value vector: \\(v_j = x_j W_V\\). This is what gets copied forward.',
  'For the focus token, build a query vector: \\(q_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_Q\\).',
  'Compute pairwise scores: \\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\).',
  'Scores are ready. We no longer need \\(Q\\) or \\(K\\) for the next step.'
];

const ATTN_QKV_COMPARE_DRAW_MS = 360;

const ATTN_QKV_COMPARE_HEAD_FADE_MS = 140;

const ATTN_QKV_COMPARE_STAGGER_MS = 190;

const ATTN_QKV_SCORE_REVEAL_DELAY_MS = 140;

const ATTN_QKV_MAX_STEP = 5;

const ATTN_STEP4_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_STEP4_FOCUS = 'sat';

const ATTN_STEP4_COMPARE_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_STEP4_SCORE_QUAL = {
  cat: { tier: 'strongest' },
  sat: { tier: 'medium' },
  on: { tier: 'medium' },
  the: { tier: 'weak' },
  mat: { tier: 'weak' }
};

const ATTN_STEP4_TAKEAWAYS = [
  'Start from Step 3 scores \\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\) for each token.',
  'Convert scores into scaled logits \\(z_j = \\frac{s_j}{\\sqrt{d_k}}\\), then normalize: \\(a_j = \\frac{\\exp(z_j)}{\\sum_{\\ell} \\exp(z_{\\ell})}\\).',
  'First multiplication: pair the first weight with its value vector and scale it.',
  'Now apply the remaining attention multiplications across the other tokens.',
  'Aggregate all weighted values: \\(o_{\\mathrm{sat}} = \\sum_j a_j v_j\\).',
  'Residual addition: \\(y_{\\mathrm{sat}} = x_{\\mathrm{sat}} + o_{\\mathrm{sat}}\\).'
];

const ATTN_STEP4_COMPARE_DRAW_MS = 360;

const ATTN_STEP4_COMPARE_HEAD_FADE_MS = 140;

const ATTN_STEP4_COMPARE_STAGGER_MS = 190;

const ATTN_STEP4_SCORE_REVEAL_DELAY_MS = 140;

const ATTN_STEP4_MAX_STEP = 5;

const ATTN_STEP4_PAIR_STAGGER_MS = 320;

const ATTN_STEP4_PAIR_ANIM_MS = 1000;

const ATTN_STEP4_PAIR_FIRST_HOLD_MS = 360;

const ATTN_STEP4_AGG_TERM_STAGGER_MS = 210;

const ATTN_STEP4_AGG_COLLAPSE_DELAY_MS = 360;

const ATTN_STEP4_AGG_COLLAPSE_MS = 340;

const ATTN_STEP4_MERGE_TARGET = 'sat';

const ATTN_STEP4_MERGE_ORDER = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_STEP4_VEC_MERGE_STAGGER_MS = 250;

const ATTN_STEP4_VEC_MERGE_MS = 760;

const ATTN_STEP4_VEC_MERGE_FADE_MS = 180;

const ATTN_STEP4_RESIDUAL_ANIM_MS = 900;

const ATTN_STEP4_RESIDUAL_FADE_MS = 180;

const ATTN_MATRIX_TOKENS = ATTN_QKV_TOKENS.slice();

const ATTN_MATRIX_TAKEAWAYS = [
  'Start from the same sequence: tokens plus their embedding rows.',
  'First collect the sequence tokens into a compact token matrix \\(T\\).',
  'Then collect the embedding rows into the embedding matrix \\(X\\).',
  'Now apply the same Step 1 projection to the whole matrix: \\(X\\) is copied into three branches and projected into \\(Q\\), \\(K\\), and \\(V\\).',
  'Bring \\(Q\\) and \\(K\\) to the center as the matrices used for score computation.',
  'Transpose \\(K\\) so the matrix dimensions line up for multiplication.',
  'Compute raw attention scores for the whole sequence: \\(S = QK^{\\mathsf{T}}\\).',
  'Without a mask, earlier tokens can attend to later tokens.<br>\\(\\mathrm{cat}\\) could use information from \\(\\mathrm{sat}\\), \\(\\mathrm{on}\\), \\(\\mathrm{the}\\), and \\(\\mathrm{mat}\\) before those tokens should exist.<br>That leaks future information and breaks left-to-right generation.',
  'The fix is a causal attention mask.<br>For every future token position, replace that score with \\(-\\infty\\) so it is blocked.<br>Only the current token and the tokens before it are allowed to remain.',
  'Notice what appears in the blocked cells: \\(-\\infty\\).<br>Why do we write negative infinity there instead of \\(0\\) or just removing the cell?<br>The next steps will make that clear when we center this matrix, scale it, and apply the row-wise softmax.',
  'Now focus on the masked score matrix itself.<br>We will operate on it row by row, because attention normalizes each query row independently.',
  'First scale the allowed scores: \\(z_{ij} = \\frac{s_{ij}}{\\sqrt{d_k}}\\).<br>The masked future positions stay at \\(-\\infty\\), so they remain blocked.<br>This keeps the score magnitudes in a stable range before softmax.',
  'Now apply softmax row by row.<br>\\(a_{ij} = \\frac{\\exp(z_{ij})}{\\sum_{\\ell=1}^{S} \\exp(z_{i\\ell})}\\), so \\(\\exp(-\\infty) = 0\\) and every masked future entry becomes zero attention.<br>Each row becomes a valid attention distribution over the allowed tokens.',
  'Bring in the value matrix \\(V\\).<br>The attention matrix \\(A\\) tells us how much of each value row to mix for every output row.',
  'Compute the weighted sum for the whole sequence: \\(O = AV\\).<br>Each output row is a weighted combination of value rows, using the attention weights from the matching row of \\(A\\).'
];

const ATTN_MATRIX_MAX_STEP = 14;

const ATTN_MATRIX_ROW_STAGGER_MS = 140;

const ATTN_MATRIX_TOKEN_TRAVEL_MS = 680;

const ATTN_MATRIX_X_TRAVEL_MS = 720;

const ATTN_MATRIX_FADE_MS = 180;

const ATTN_MATRIX_PROJS = ['q', 'k', 'v'];

const ATTN_MATRIX_PROJ_LABELS = { q: 'W_Q', k: 'W_K', v: 'W_V' };

const ATTN_MATRIX_OUTPUT_LABELS = { q: 'Q', k: 'K', v: 'V' };

const ATTN_MATRIX_Q_VECTORS = {
  cat: [0.8, 0.6, 0.2, 0.2],
  sat: ATTN_QKV_QUERY_VECTOR.slice(),
  on: [0.5, 0.7, 0.6, 0.3],
  the: [0.2, 0.5, 0.6, 0.4],
  mat: [0.2, 0.3, 0.8, 0.7]
};

const ATTN_MATRIX_PROJ_STAGGER_MS = 150;

const ATTN_MATRIX_PROJ_ANIM_MS = 260;

const ATTN_MATRIX_PROJ_FADE_MS = 200;

const ATTN_MATRIX_D_K = ATTN_QKV_QUERY_VECTOR.length;

const ATTN_MATRIX_D_V = (ATTN_QKV_VALUE_VECTORS[ATTN_MATRIX_TOKENS[0]] || []).length;

const ATTN_MATRIX_SCORE_TOKENS = ATTN_MATRIX_TOKENS.slice();

const ATTN_MATRIX_SCORE_CLEANUP_MS = 280;

const ATTN_MATRIX_SCORE_CENTER_MS = 620;

const ATTN_MATRIX_SCORE_TRANSPOSE_MS = 360;

const ATTN_MATRIX_SCORE_ROW_STAGGER_MS = 96;

const ATTN_MATRIX_SCORE_FADE_MS = 200;

const ATTN_MATRIX_MASK_ROW_STAGGER_MS = 120;

const ATTN_MATRIX_MASK_FADE_MS = 220;

const ATTN_MATRIX_POSTSCORE_CENTER_MS = 620;

const ATTN_MATRIX_SCALE_FADE_MS = 220;

const ATTN_MATRIX_SOFTMAX_ROW_STAGGER_MS = 140;

const ATTN_MATRIX_SOFTMAX_ROW_MS = 240;

const ATTN_MATRIX_VALUE_ENTRY_MS = 420;

const ATTN_MATRIX_OUTPUT_ROW_STAGGER_MS = 140;

const ATTN_MATRIX_OUTPUT_ROW_MS = 240;

const ATTN_MHA_TAKEAWAYS = [
  'Multi-head attention runs several smaller attention mechanisms in parallel on different learned projections of the same embedding matrix \\(X\\).',
  'Split the same embedding matrix \\(X\\) into two parallel heads. Both heads see the full sequence, but each head will work in its own smaller learned subspace.',
  'Once both heads have their own copy of \\(X\\), the shared source matrix is no longer needed on-screen. Remove it and use the space to focus on the per-head computation.',
  'Each head applies its own learned projection matrices \\(W_Q\\), \\(W_K\\), and \\(W_V\\) to the same \\(X\\), producing different head-specific \\(Q\\), \\(K\\), and \\(V\\).',
  'Each head now runs its own masked attention mechanism in parallel. Because the heads use different learned projections, they can focus on different relationships or features in the same sequence.',
  'Instead of one attention-based summary, the model now has multiple context representations, one per head.',
  'Bring the head outputs together and concatenate them side by side. This combines the separate head-specific context representations into one wider sequence matrix.',
  'Apply the output projection \\(W_O\\) to mix the concatenated head information back into one shared representation. This is the final multi-head attention output.'
];

const ATTN_MHA_TOKENS = ATTN_MATRIX_TOKENS.slice();

const ATTN_MHA_HEADS = ['h1', 'h2'];

const ATTN_MHA_HEAD_LABELS = { h1: 'Head 1', h2: 'Head 2' };

const ATTN_MHA_HEAD_DIM = 2;

const ATTN_MHA_MODEL_DIM = 4;

const ATTN_MHA_COMBINED_DIM = ATTN_MHA_HEADS.length * ATTN_MHA_HEAD_DIM;

const ATTN_MHA_WO_DIMS = { rows: 4, cols: 4 };

const ATTN_MHA_HEAD_SLICES = { h1: [0, 1], h2: [2, 3] };

const ATTN_MHA_MAX_STEP = 7;

const ATTN_MHA_PROJS = ['q', 'k', 'v'];

const ATTN_MHA_SPLIT_MS = 620;

const ATTN_MHA_PROJ_STAGGER_MS = 120;

const ATTN_MHA_PROJ_REVEAL_MS = 220;

const ATTN_MHA_ATTN_STAGGER_MS = 110;

const ATTN_MHA_ATTN_REVEAL_MS = 220;

const ATTN_MHA_OUTPUT_ROW_STAGGER_MS = 140;

const ATTN_MHA_OUTPUT_ROW_MS = 220;

const ATTN_MHA_CONCAT_MS = 680;

const ATTN_MHA_WO_REVEAL_MS = 260;

const ATTN_POS_MAX_STEP = 3;

const ATTN_POS_SEQ_A = ['cat', 'sat', 'on', 'mat'];

const ATTN_POS_SEQ_B = ['mat', 'sat', 'on', 'cat'];

const ATTN_POS_TAGS = ['p1', 'p2', 'p3', 'p4'];

const ATTN_POS_TAKEAWAYS = [
  'RNNs read tokens one at a time, so sequence order is naturally built into the computation.',
  'Transformers process all token rows together. If everything enters attention at once, where does the sequence information come from?',
  'Without positional embeddings, the model treats the sequence like a bag of tokens. Self-attention is order-blind: reordering the same token rows does not tell it what came first.',
  'Positional embeddings fix this. Inject a different position signal pointwise into each token embedding before attention, so the model can tell first, second, third, and fourth apart.'
];

const ATTN_P1_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_P1_FOCUS = 'sat';

const ATTN_P1_PROJS = ['q', 'k', 'v'];

const ATTN_P1_PROJ_NAMES = { q: 'Query (Q)', k: 'Key (K)', v: 'Value (V)' };

const ATTN_P1_MEANINGS = { q: 'what I need', k: 'what I offer', v: 'what I pass forward' };

const ATTN_P1_MAX_STEP = 8;

const ATTN_P1_TAKEAWAYS = [
  'Attention turns each token state into three role-specific versions.',
  'Start from the current token state for \\(\\mathrm{sat}\\): \\(x_{\\mathrm{sat}}\\).',
  'Copy \\(x_{\\mathrm{sat}}\\) into three branches: one each for Q, K, and V.',
  'Apply the query matrix \\(W_Q\\) to get \\(q_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_Q\\).',
  'Query \\(q_{\\mathrm{sat}}\\): what information does \\(\\mathrm{sat}\\) need?',
  'Apply the key matrix \\(W_K\\) to get \\(k_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_K\\).',
  'Key \\(k_{\\mathrm{sat}}\\): what does \\(\\mathrm{sat}\\) offer to other tokens?',
  'Apply the value matrix \\(W_V\\) to get \\(v_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_V\\).',
  'Next: use \\(q_{\\mathrm{sat}}\\) to score every \\(k_j\\), producing \\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\).'
];

const ATTN_WGT_TOKENS = ATTN_QKV_TOKENS.slice();

const ATTN_WGT_FOCUS = 'sat';

const ATTN_WGT_D_K = ATTN_QKV_QUERY_VECTOR.length;

const ATTN_WGT_MAX_STEP = 3;

const ATTN_WGT_TAKEAWAYS = [
  'Raw similarity scores between \\(q_{\\mathrm{sat}}\\) and each key \\(k_j\\), producing \\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\).',
  'Scale each score into \\(z_j = \\frac{s_j}{\\sqrt{d_k}}\\) to keep values in a stable range before \\(\\operatorname{softmax}\\).',
  'Normalize row-wise: \\(a_j = \\frac{\\exp(z_j)}{\\sum_{\\ell} \\exp(z_{\\ell})}\\).',
  'These weights decide how much each value \\(v_j\\) contributes to the updated token.'
];

let ATTN_MATRIX_ATTN_ROWS;
let ATTN_MATRIX_CAUSAL_MASK;
let ATTN_MATRIX_OUTPUT_ROWS;
let ATTN_MATRIX_SCORE_MASKED_ROWS;
let ATTN_MATRIX_SCORE_RAW_MASKED_ROWS;
let ATTN_MATRIX_SCORE_RAW_ROWS;
let ATTN_MATRIX_SCORE_SCALED_ROWS;
let ATTN_MHA_ATTN_ROWS;
let ATTN_MHA_K_ROWS;
let ATTN_MHA_MASKED_ROWS;
let ATTN_MHA_OUTPUT_ROWS;
let ATTN_MHA_Q_ROWS;
let ATTN_MHA_SCORE_ROWS;
let ATTN_MHA_V_ROWS;
let ATTN_STEP4_AGG_VECTOR;
let ATTN_STEP4_RESIDUAL_INPUT_VECTOR;
let ATTN_STEP4_RESIDUAL_OUTPUT_VECTOR;
let ATTN_STEP4_SCORE_BY_TOKEN;
let ATTN_STEP4_WEIGHTED_VALUE_VECTORS;
let ATTN_STEP4_WEIGHT_BY_TOKEN;
let ATTN_WGT_RAW_SCORES;
let ATTN_WGT_RAW_SCORE_BY_TOKEN;
let ATTN_WGT_SCALED_SCORES;
let ATTN_WGT_WEIGHTS;
