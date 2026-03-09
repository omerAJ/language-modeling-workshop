function typesetMath(root = document) {
  if (!window.MathJax || !window.MathJax.typesetPromise) return;
  window.MathJax.typesetPromise([root]).catch(() => {});
}
