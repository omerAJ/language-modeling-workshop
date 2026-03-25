const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const SLIDE_ORDER = [0, 1, 2, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 8, 22, 23, 24, 16, 29, 30, 31, 32, 33];

const VIDEO_CLIP_SLIDE_IDS = [2, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50];

const ABS_CLIP_FALLBACK_DIR = '/home/maincoder/Documents/inside-LLM/manimations/media/videos/pipeline/1080p60/LanguageModelingPipeline_snippets';

const INGREDIENT_COLORS = {
  1: 'rgba(0,255,255,0.35)',
  2: 'rgba(108,140,255,0.35)',
  3: 'rgba(167,139,250,0.4)'
};

const INF_TOKENS = ['Islamabad', '.', '⟨EOS⟩'];

const TRAIN_EXPLANATIONS = {
  1: '<strong>Forward pass</strong> — identical to inference. Tokens go in, the Transformer processes them through all its layers, and a probability distribution over possible next tokens comes out. Nothing is different yet.',
  2: '<strong>The teacher signal</strong> — during training, we already know the correct next token. It\'s simply the next word in the training data. This ground truth is what makes learning possible — the model doesn\'t have to figure out what\'s right on its own.',
  3: '<strong>Cross-entropy loss</strong> — measures how much probability the model gave to the correct token. High confidence in the right answer \u2192 low loss. Surprised by it \u2192 high loss. Formally: \\(\\mathcal{L} = -\\log P(\\text{correct token})\\).',
  4: '<strong>Backpropagation</strong> — the chain rule traces the loss backward through every layer. For each of the billions of weights, it answers: \"how much did <em>you</em> contribute to the error?\" This produces a gradient \\(\\frac{\\partial \\mathcal{L}}{\\partial \\theta}\\) for every single parameter.',
  5: '<strong>Weight update</strong> — each weight is nudged in the direction that reduces the loss. The learning rate \\(\\eta\\) controls step size: \\(\\theta \\leftarrow \\theta - \\eta \\cdot \\nabla \\mathcal{L}\\). After updating, the next batch runs through a slightly better model. Repeat billions of times.'
};

const VOCAB = [
  { word: 'Islamabad', predicted: 0.08, target: 1.00, after: 0.28 },
  { word: 'Karachi',   predicted: 0.11, target: 0.00, after: 0.06 },
  { word: 'the',       predicted: 0.14, target: 0.00, after: 0.09 },
  { word: 'Lahore',    predicted: 0.06, target: 0.00, after: 0.03 },
  { word: 'a',         predicted: 0.10, target: 0.00, after: 0.07 },
  { word: 'Peshawar',  predicted: 0.04, target: 0.00, after: 0.03 },
  { word: 'Delhi',     predicted: 0.05, target: 0.00, after: 0.02 },
  { word: 'not',       predicted: 0.07, target: 0.00, after: 0.05 },
  { word: '...',       predicted: 0.35, target: 0.00, after: 0.37 },
];

const INITIAL_DRAMA_RESULT_HTML = '<span class="icon">?</span><span>Select a suspect. Then check whether your choice satisfies the timing + access constraints.</span>';
