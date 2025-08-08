const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const previewImg = document.getElementById('previewImg');
const resultEl = document.getElementById('result');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

predictBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    resultEl.textContent = 'Please choose an image.';
    return;
  }
  resultEl.textContent = 'Predicting...';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
      resultEl.textContent = `Error: ${err.detail || resp.statusText}`;
      return;
    }

    const data = await resp.json();
    const label = data.predicted_class;
    const conf = (data.confidence * 100).toFixed(1);

    resultEl.innerHTML = `<strong>Prediction:</strong> ${label} <br/><strong>Confidence:</strong> ${conf}%`;
  } catch (e) {
    console.error(e);
    resultEl.textContent = 'Network error. Is the API running?';
  }
}); 