const ARCH3_MAX_STEP = 2;
const ARCH3_EQUATION_HTML = '\\(\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^{\\mathsf{T}}}{\\sqrt{d_k}}\\right)V\\)';

function ensureArchitectureOverviewEquation(slide) {
  const equationShell = slide ? slide.querySelector('.arch3-equation-shell') : null;
  const equationMain = document.getElementById('arch3-equation-main');
  if (!equationShell || !equationMain) return;
  if (equationShell.dataset.mathReady === 'true' || equationShell.dataset.mathReady === 'loading') return;

  equationShell.dataset.mathReady = 'loading';
  setMathHTML(equationMain, ARCH3_EQUATION_HTML).then(() => {
    equationShell.dataset.mathReady = 'true';
    scheduleDeckRefresh({ reason: 'slide-3-equation', typeset: false });
  });
}

function setArchitectureOverviewStep(step) {
  const slide = document.getElementById('slide-3');
  if (!slide) return;

  const blockShell = slide.querySelector('.arch3-block-shell');
  const equationShell = slide.querySelector('.arch3-equation-shell');
  const clamped = Math.max(0, Math.min(ARCH3_MAX_STEP, step));

  architectureOverviewState.step = clamped;
  slide.classList.toggle('arch3-show-block', clamped >= 1);
  slide.classList.toggle('arch3-show-eq', clamped >= 2);

  if (blockShell) blockShell.setAttribute('aria-hidden', clamped >= 1 ? 'false' : 'true');
  if (equationShell) equationShell.setAttribute('aria-hidden', clamped >= 2 ? 'false' : 'true');

  if (clamped >= 2) {
    requestAnimationFrame(() => ensureArchitectureOverviewEquation(slide));
  }
}

function initArchitectureOverviewSlide() {
  architectureOverviewState.initialized = true;
  setArchitectureOverviewStep(architectureOverviewState.step || 0);
}

function runArchitectureOverviewStep() {
  if (architectureOverviewState.step >= ARCH3_MAX_STEP) return false;
  setArchitectureOverviewStep(architectureOverviewState.step + 1);
  return true;
}

function resetArchitectureOverviewSlide() {
  setArchitectureOverviewStep(0);
}
