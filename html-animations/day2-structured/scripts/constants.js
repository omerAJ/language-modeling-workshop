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

const ATTN_INTRO_FOOTER = 'Attention now needs a scoring rule: what matters, and what gets copied forward?';

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
  'Start from embeddings \\(x_1, \\ldots, x_n\\).',
  'Project each token into a key vector: \\(k_j = x_j W_K\\).',
  'Project each token into a value vector: \\(v_j = x_j W_V\\). These are the vectors copied forward.',
  'For the focus token, build a query vector: \\(q_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_Q\\).',
  'Score each token by comparing \\(q_{\\mathrm{sat}}\\) with \\(k_j\\): \\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\).',
  'The scores are ready. The next step turns them into attention weights.'
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
  'Start from the Step 3 scores for \\(q_{\\mathrm{sat}}\\).',
  'Convert them into attention weights \\(a_j\\).',
  'Apply the first weight to its value vector.',
  'Apply the remaining weights to the other value vectors.',
  'Sum the weighted values into \\(o_{\\mathrm{sat}}\\).',
  'Add the residual: \\(x\'_{\\mathrm{sat}} = x_{\\mathrm{sat}} + o_{\\mathrm{sat}}\\).'
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
  'Start from the same sequence: tokens and their embedding rows.',
  'Collect the sequence into a compact token matrix \\(T\\).',
  'Collect the embedding rows into the embedding matrix \\(X\\).',
  'Lets collapse the original sequence view and keep the compact matrices.',
  'Project \\(X\\) into \\(Q\\), \\(K\\), and \\(V\\) for the full sequence at once.',
  'Bring \\(Q\\) and \\(K\\) to the center as the matrices used for score computation.',
  'Transpose \\(K\\) so the matrix dimensions line up for multiplication.',
  'Compute raw attention scores for the whole sequence: \\(S = QK^{\\mathsf{T}}\\).',
  'Without a causal mask, earlier tokens can attend to later tokens.<br>That leaks future information and breaks left-to-right generation.',
  'The fix is a causal mask.<br>Replace every future-position score with \\(-\\infty\\), leaving only the current and earlier tokens available.',
  'Blocked cells show \\(-\\infty\\), not \\(0\\).<br>Softmax will turn those entries into exactly zero attention.',
  'Focus on the masked score matrix row by row.<br>Attention normalizes each query row independently.',
  'Scale the allowed scores: \\(z_{ij} = \\frac{s_{ij}}{\\sqrt{d_k}}\\).<br>Masked future positions stay at \\(-\\infty\\).',
  'Apply softmax row by row.<br>Because \\(\\exp(-\\infty) = 0\\), masked future entries become zero attention.<br>Each row becomes a valid distribution.',
  'Bring in the value matrix \\(V\\).<br>The attention matrix \\(A\\) determines how much of each value row to mix into each output row.',
  'Compute the weighted sum for the full sequence: \\(O = AV\\).<br>Each output row is a weighted combination of value rows.'
];

const ATTN_MATRIX_MAX_STEP = 15;

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
  '',
  'Make two copies of \\(X\\), one for each head, so both attention paths can run in parallel.',
  'Each head starts from the same input sequence \\(X\\), then computes attention in its own learned subspace.',
  'Each head applies its own learned projection matrices \\(W_Q\\), \\(W_K\\), and \\(W_V\\) to the same \\(X\\), producing different head-specific \\(Q\\), \\(K\\), and \\(V\\).',
  'Each head now runs masked attention in parallel. Different learned projections let different heads focus on different relationships in the same sequence.',
  'The model now has one context representation per head.',
  'Concatenate the head outputs side by side to form one wider sequence matrix.',
  'Apply the output projection \\(W_O\\) to mix the concatenated head outputs back into one shared representation.'
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

/** Multi-head split: gap between chained bus strokes (same dash-draw feel as slide 20 compare). */
const ATTN_MHA_BUS_SEGMENT_GAP_MS = 48;

const ATTN_MHA_PROJ_STAGGER_MS = 120;

const ATTN_MHA_PROJ_REVEAL_MS = 220;

/** Delay after projection opens before showing W and ×, = (per head, after stagger). */
const ATTN_MHA_PROJ_DUMMY_MS = 260;

/** Delay after weights appear before revealing Q/K/V result matrices. */
const ATTN_MHA_PROJ_WEIGHTS_MS = 280;

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
  'Without positional embeddings, self-attention is permutation equivariant. Reorder the token rows, and it produces the same comparisons with the outputs reordered the same way, so nothing in the layer marks which token was first.',
  'Positional embeddings fix this. Inject a different position signal pointwise into each token embedding before attention, so the model can tell first, second, third, and fourth apart.'
];

const ATTN_P1_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];

const ATTN_P1_FOCUS = 'sat';

const ATTN_P1_PROJS = ['q', 'k', 'v'];

const ATTN_P1_PROJ_NAMES = { q: 'Query (Q)', k: 'Key (K)', v: 'Value (V)' };

const ATTN_P1_MEANINGS = { q: 'what this token seeks', k: 'what this token offers', v: 'what this token carries' };

const ATTN_P1_MAX_STEP = 8;

const ATTN_P1_TAKEAWAYS = [
  'Attention creates three role-specific versions of each token state.',
  'Start from the current token state for \\(\\mathrm{sat}\\): \\(x_{\\mathrm{sat}}\\).',
  'Copy \\(x_{\\mathrm{sat}}\\) into three branches: one each for Q, K, and V.',
  'Apply the query matrix \\(W_Q\\) to get \\(q_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_Q\\).',
  'Query \\(q_{\\mathrm{sat}}\\): what this token seeks.',
  'Apply the key matrix \\(W_K\\) to get \\(k_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_K\\).',
  'Key \\(k_{\\mathrm{sat}}\\): what this token offers other tokens.',
  'Apply the value matrix \\(W_V\\) to get \\(v_{\\mathrm{sat}} = x_{\\mathrm{sat}} W_V\\).',
  'Next, score every \\(k_j\\) against \\(q_{\\mathrm{sat}}\\): \\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\).'
];

const ATTN_WGT_TOKENS = ATTN_QKV_TOKENS.slice();

const ATTN_WGT_FOCUS = 'sat';

const ATTN_WGT_D_K = ATTN_QKV_QUERY_VECTOR.length;

const ATTN_WGT_MAX_STEP = 3;

const ATTN_WGT_TAKEAWAYS = [
  'Raw similarity scores compare \\(q_{\\mathrm{sat}}\\) with each key \\(k_j\\): \\(s_j = q_{\\mathrm{sat}}^{\\mathsf{T}} k_j\\).',
  'Scale each score into \\(z_j = \\frac{s_j}{\\sqrt{d_k}}\\) to keep magnitudes stable before \\(\\operatorname{softmax}\\).',
  'Normalize row-wise: \\(a_j = \\frac{\\exp(z_j)}{\\sum_{\\ell} \\exp(z_{\\ell})}\\).',
  'These weights determine how much each value vector \\(v_j\\) contributes to the update.'
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
