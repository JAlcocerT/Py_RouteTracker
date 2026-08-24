import { useState } from 'react'
import './App.css'
import { ConfigurePage } from './pages/ConfigurePage'
import { RenderPage } from './pages/RenderPage'
import { UploadPage } from './pages/UploadPage'
import { InstallPrompt } from './pwa/InstallPrompt'
import { OfflineBanner } from './pwa/OfflineBanner'
import { UpdateToast } from './pwa/UpdateToast'

type Step =
  | { name: 'upload' }
  | { name: 'configure'; videoId: string; videoFile: File }
  | { name: 'render'; jobId: string; videoId: string; claimToken: string }

function App() {
  const [step, setStep] = useState<Step>({ name: 'upload' })

  return (
    <>
      <OfflineBanner />
      {(() => {
        switch (step.name) {
          case 'upload':
            return <UploadPage onUploaded={(videoId, videoFile) => setStep({ name: 'configure', videoId, videoFile })} />
          case 'configure':
            return (
              <ConfigurePage
                videoId={step.videoId}
                videoFile={step.videoFile}
                onRenderStarted={(jobId, claimToken) => setStep({ name: 'render', jobId, videoId: step.videoId, claimToken })}
              />
            )
          case 'render':
            return (
              <RenderPage
                jobId={step.jobId}
                videoId={step.videoId}
                claimToken={step.claimToken}
                onStartOver={() => setStep({ name: 'upload' })}
              />
            )
        }
      })()}
      <InstallPrompt />
      <UpdateToast />
    </>
  )
}

export default App
