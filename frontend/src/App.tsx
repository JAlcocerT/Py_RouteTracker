import { useState } from 'react'
import './App.css'
import { ConfigurePage } from './pages/ConfigurePage'
import { RenderPage } from './pages/RenderPage'
import { UploadPage } from './pages/UploadPage'

type Step =
  | { name: 'upload' }
  | { name: 'configure'; videoId: string; videoFile: File }
  | { name: 'render'; jobId: string; videoId: string }

function App() {
  const [step, setStep] = useState<Step>({ name: 'upload' })

  switch (step.name) {
    case 'upload':
      return <UploadPage onUploaded={(videoId, videoFile) => setStep({ name: 'configure', videoId, videoFile })} />
    case 'configure':
      return (
        <ConfigurePage
          videoId={step.videoId}
          videoFile={step.videoFile}
          onRenderStarted={(jobId) => setStep({ name: 'render', jobId, videoId: step.videoId })}
        />
      )
    case 'render':
      return <RenderPage jobId={step.jobId} videoId={step.videoId} onStartOver={() => setStep({ name: 'upload' })} />
  }
}

export default App
