function typesetMath(root = document) {
  if (!window.MathJax || !window.MathJax.typesetPromise) return Promise.resolve();
  return window.MathJax.typesetPromise([root]).catch(() => {});
}

function clearTypesetMath(root = document) {
  if (!window.MathJax || !window.MathJax.typesetClear) return;
  window.MathJax.typesetClear([root]);
}

function setMathHTML(el, html) {
  if (!el) return Promise.resolve();
  clearTypesetMath(el);
  el.innerHTML = html;
  return typesetMath(el);
}

function setMathText(el, tex) {
  return setMathHTML(el, inlineMath(tex));
}
