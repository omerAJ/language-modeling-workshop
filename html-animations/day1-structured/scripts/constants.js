const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const SLIDE_ORDER = [0, 1, 3, 4, 5, 6, 7, '7b', 9, 10, 11, 12, 13, 14, 8, 22, 23, 24, 16, 29, 30, 31, 32, 33, '32b'];

const INGREDIENT_COLORS = {
  1: 'rgba(0,255,255,0.35)',
  2: 'rgba(108,140,255,0.35)',
  3: 'rgba(167,139,250,0.4)'
};

const INF_TOKENS = ['Islamabad', '.', '⟨EOS⟩'];

const TRAIN_INTRO_HTML = '<strong>Reveal the loop in order.</strong> Start with the forward pass, then compare with the correct token, compute loss, backpropagate error, and update the weights.';

const TRAIN_EXPLANATIONS = {
  1: '<strong>Forward pass</strong> — this part is identical to inference: context goes in, next-token scores come out.',
  2: '<strong>Ground truth</strong> — training already knows the correct next token from the dataset, so the model can compare prediction vs. answer.',
  3: '<strong>Cross-entropy loss</strong> — if the model gave low probability to the correct token, the loss is high: \\(\\mathcal{L} = -\\log P(\\text{correct token})\\).',
  4: '<strong>Backpropagation</strong> — the chain rule sends that error backward and computes a gradient for every weight.',
  5: '<strong>Weight update</strong> — move each weight a little in the direction that reduces loss: \\(\\theta \\leftarrow \\theta - \\eta \\cdot \\nabla \\mathcal{L}\\).'
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

const INITIAL_DRAMA_RESULT_HTML = '<span class="icon">?</span><span>Pick the suspect whose timeline fits every clue.</span>';
