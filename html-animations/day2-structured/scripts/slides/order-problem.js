const ORDER25_MAX_STEP = 4;
const ORDER25_CLASSES = [
  'order25-show-left',
  'order25-show-right',
  'order25-show-update',
  'order25-show-proof'
];
const ORDER25_TAKEAWAYS = [
  'Without position, attention only sees token rows. Reordering the rows changes where things appear, not which token vectors exist.',
  'Fix the query to dog: q_dog = [2,1] and the original key order produce the score row [5, 4, 2].',
  'After permutation, the same dot products reappear in a new order: [5, 4, 2] becomes [2, 4, 5]. The computation is reordered, not changed.',
  'This is why it matters: the weights and value terms permute together, so the dog token receives the same attention update in both sequences.',
  "The example showed the same update to the dog row. The proof shows this always happens: O' = PO, so order only appears once positional information is added."
];

function setOrderProblemStep(step) {
  const slide = document.getElementById('slide-25');
  const takeaway = document.getElementById('order25-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(ORDER25_MAX_STEP, step));
  orderProblemState.step = clamped;
  ORDER25_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.textContent = ORDER25_TAKEAWAYS[clamped] || ORDER25_TAKEAWAYS[0];
}

function initOrderProblemSlide() {
  orderProblemState.initialized = true;
  setOrderProblemStep(orderProblemState.step || 0);
}

function runOrderProblemStep() {
  if (!orderProblemState.initialized) initOrderProblemSlide();
  if (orderProblemState.step >= ORDER25_MAX_STEP) return false;
  setOrderProblemStep(orderProblemState.step + 1);
  return true;
}

function resetOrderProblemSlide() {
  setOrderProblemStep(0);
}
