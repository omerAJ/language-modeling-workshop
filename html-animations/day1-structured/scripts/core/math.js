function typesetMath(root = document) {
  if (!window.MathJax || !window.MathJax.typesetPromise) return Promise.resolve();
  return window.MathJax.typesetPromise([root]).catch(() => {});
}
