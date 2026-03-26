/* Slide 25: Shuffle the Words, Break the Meaning.
   Pure autostep — no JS step controller needed.
   Keep init/reset stubs for the registry. */

function initOrderProblemSlide() {
  orderProblemState.initialized = true;
}

function runOrderProblemStep() {
  return false;
}

function resetOrderProblemSlide() {
  orderProblemState.initialized = false;
}
