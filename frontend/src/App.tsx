import { useState } from 'react'
import './App.css'
import { ConfigurePage } from './pages/ConfigurePage'
import { RenderPage } from './pages/RenderPage'
import { UploadPage } from './pages/UploadPage'
import { InstallPrompt } from './pwa/InstallPrompt'
import { UpdateToast } from './pwa/UpdateToast'
import type { AnnotatedRow } from './lib/laps/detection'
import type { RenderStyle, TelemetryPoint, WidgetSelection } from './types'

type Step =
  | { name: 'upload' }
  | { name: 'configure'; videoFile: File; duration: number; telemetry: TelemetryPoint[]; hasAccel: boolean }
  | {
      name: 'render'
      videoFile: File
      trimStart: number
      trimEnd: number
      widgets: WidgetSelection
      style: RenderStyle
      annotatedRows: AnnotatedRow[]
    }

function App() {
  const [step, setStep] = useState<Step>({ name: 'upload' })

  return (
    <>
      {(() => {
        switch (step.name) {
          case 'upload':
            return (
              <UploadPage
                onUploaded={(videoFile, duration, telemetry, hasAccel) =>
                  setStep({ name: 'configure', videoFile, duration, telemetry, hasAccel })
                }
              />
            )
          case 'configure':
            return (
              <ConfigurePage
                videoFile={step.videoFile}
                duration={step.duration}
                telemetry={step.telemetry}
                hasAccel={step.hasAccel}
                onRenderStarted={(payload) => setStep({ name: 'render', ...payload })}
              />
            )
          case 'render':
            return (
              <RenderPage
                videoFile={step.videoFile}
                trimStart={step.trimStart}
                trimEnd={step.trimEnd}
                widgets={step.widgets}
                style={step.style}
                annotatedRows={step.annotatedRows}
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
