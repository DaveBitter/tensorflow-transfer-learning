import { useRef, useState, useEffect } from "react";

export const useWebcam = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoPlaying, setVideoPlaying] = useState(false);

  const toggleWebcam = async () => {
    if (!videoRef.current) {
      return;
    }

    if (videoPlaying) {
      videoRef.current.srcObject?.getVideoTracks().forEach((track) => {
        track.stop();
      });
      setVideoPlaying(false);
      return;
    }

    if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
      // getUsermedia parameters.
      const constraints = {
        video: true,
        width: 640,
        height: 480,
      };

      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        if (!videoRef.current) {
          return;
        }
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener("loadeddata", function () {
          setVideoPlaying(true);
        });
      });
    } else {
      console.warn("getUserMedia() is not supported by your browser");
    }
  };

  useEffect(() => {
    navigator.permissions
      .query({ name: "camera" as PermissionName })
      .then(function (permissionStatus) {
        if (permissionStatus.state === "granted") {
          toggleWebcam();
        }
      });
  }, []);

  return {
    toggleWebcam,
    videoRef,
    videoPlaying,
  };
};
