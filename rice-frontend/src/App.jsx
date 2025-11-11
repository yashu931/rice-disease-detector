import React, {useState, useRef} from 'react'

export default function App(){
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [resultHTML, setResultHTML] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const uploadSectionRef = useRef(null)

  function onFilesSelected(files){
    if(!files || files.length === 0) return
    const f = files[0]
    if(!f.type.startsWith('image/')){
      setError('Please select an image file (jpg/png).')
      return
    }
    setError(null)
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
    setResultHTML(null)
  }

  async function handleUpload(){
    if(!file){ setError('No file selected'); return }
    setUploading(true)
    try{
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch('/upload', { method: 'POST', body: fd })
      if(!res.ok) throw new Error('Server error ' + res.status)
      const html = await res.text()
      setResultHTML(html)
    }catch(e){ setError(e.message) }finally{ setUploading(false) }
  }

  function scrollToUpload(){
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="flex flex-col items-center bg-gradient-to-b from-emerald-50 to-white min-h-screen">
      {/* ---------- HERO SECTION ---------- */}
      <section className="w-full flex flex-col items-center justify-center text-center py-20 px-4 bg-gradient-to-r from-[#2D5016] to-emerald-600 text-white">
        <h1 className="text-3xl md:text-5xl font-bold mb-4">AI-Powered Rice Leaf Disease Detection</h1>
        <p className="max-w-xl text-lg opacity-90 mb-8">
          Detect rice leaf diseases instantly using AI — upload a leaf image to get predictions and treatment advice in real time.
        </p>
        <button
          onClick={scrollToUpload}
          className="bg-white text-[#2D5016] font-semibold px-6 py-3 rounded-full shadow hover:scale-105 transition-transform"
        >
          Upload Leaf Image
        </button>
      </section>

      {/* ---------- UPLOAD SECTION ---------- */}
      <section ref={uploadSectionRef} className="w-full max-w-2xl p-4 mt-10">
        <div className="bg-white p-5 rounded-xl shadow-md border border-emerald-100">
          <h2 className="text-xl font-semibold text-[#2D5016] text-center mb-4">Upload Your Leaf Image</h2>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={e => onFilesSelected(e.target.files)}
            className="hidden"
          />
          {!preview ? (
            <div
              className="border-2 border-dashed border-emerald-300 rounded-lg p-8 text-center cursor-pointer hover:bg-emerald-50 transition"
              onClick={() => inputRef.current.click()}
            >
              <p className="text-slate-500">Click or drop an image here</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <img src={preview} alt="preview" className="max-h-64 rounded-md" />
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="bg-[#2D5016] text-white px-4 py-2 rounded-md hover:bg-emerald-800 transition"
              >
                {uploading ? 'Uploading...' : 'Upload & Predict'}
              </button>
            </div>
          )}
          {error && <div className="mt-3 text-red-600 text-sm text-center">{error}</div>}
        </div>

        {/* ---------- RESULTS SECTION ---------- */}
        <div className="w-full mt-8">
          {resultHTML ? (
            <div dangerouslySetInnerHTML={{ __html: resultHTML }} />
          ) : (
            <p className="text-slate-500 text-sm text-center">Results will appear here after upload.</p>
          )}
        </div>
      </section>

      {/* ---------- FOOTER ---------- */}
      <footer className="mt-16 py-6 text-center text-sm text-slate-500">
        🌾 Developed with ❤️ using Flask, TensorFlow & React
      </footer>
    </div>
  )
}
