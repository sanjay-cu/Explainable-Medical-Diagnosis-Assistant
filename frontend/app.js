javascript
async function send(){
  const file = document.getElementById('img').files[0];
  if(!file){ alert('select image'); return }
  const fd = new FormData();
  fd.append('image', file);
  const res = await fetch('/api/predict', {method:'POST', body: fd});
  const j = await res.json();
  document.getElementById('out').innerText = JSON.stringify(j, null, 2);
}

