/** Reads a video File's duration client-side via a throwaway <video>
 * element -- replaces the server's ffprobe-based `get_video_duration`
 * (app/core/ffmpeg_utils.py). No mp4box/demux needed for this alone. */
export function probeVideoDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(url)
      resolve(video.duration)
    }
    video.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error(`Could not read video metadata for '${file.name}'`))
    }
    video.src = url
  })
}
