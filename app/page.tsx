"use client";
import { useMobileNet } from "@/src/hooks/useMobileNet";
import { useTensorFlow } from "@/src/hooks/useTensorFlow";
import { useWebcam } from "@/src/hooks/useWebcam";
import { CameraIcon, EyeOpenIcon, ResetIcon } from "@radix-ui/react-icons";
import { Button, Card, Heading, Switch, Text, Spinner } from "@radix-ui/themes";
import { useRef, useState } from "react";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const numberOfClasses = 3;
export default function Home() {
  const { toggleWebcam, videoPlaying, videoRef } = useWebcam();

  const tf = useTensorFlow();
  const { model, mobileNet, readyToTrainAndPredict } = useMobileNet({
    tf,
    numberOfClasses,
  });

  const [progressBarValues, setProgressBarValues] = useState(
    Array(numberOfClasses).fill(0)
  );
  const shouldPredict = useRef(false);

  function predictLoop() {
    tf?.tidy(function () {
      if (!shouldPredict.current || !tf || !mobileNet || !model) {
        return;
      }

      let videoFrameAsTensor = tf.browser.fromPixels(videoRef.current).div(255);
      let resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [224, 224],
        true
      );

      let imageFeatures = mobileNet.predict(resizedTensorFrame.expandDims());
      let prediction = model.predict(imageFeatures).squeeze();
      let predictionArray = prediction.arraySync();

      setProgressBarValues(predictionArray);
      console.log(predictionArray);
    });

    window.requestAnimationFrame(predictLoop);
  }

  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState<
    {
      input: any;
      output: number;
    }[]
  >([]);

  const trainAndPredict = async () => {
    setIsTraining(true);

    const xs = tf.stack(trainingData.map((data) => data.input));
    const ys = tf.oneHot(
      trainingData.map((data) => data.output),
      numberOfClasses
    );

    await model.fit(xs, ys, {
      epochs: 20,
    });

    setIsTraining(false);
    shouldPredict.current = true;
    predictLoop();
  };

  const [capturedImages, setCapturedImages] = useState<{
    [key: string]: string[];
  }>(
    Array.from({ length: numberOfClasses }).reduce(
      (
        acc: {
          [key: string]: string[];
        },
        _,
        i
      ) => ({ ...acc, [`class_${i + 1}`]: [] }),
      {}
    )
  );

  const reset = () => {
    shouldPredict.current = false;
    setTrainingData([]);
    setCapturedImages(
      Array.from({ length: numberOfClasses }).reduce(
        (
          acc: {
            [key: string]: string[];
          },
          _,
          i
        ) => ({ ...acc, [`class_${i + 1}`]: [] }),
        {}
      )
    );
    setProgressBarValues(Array(numberOfClasses).fill(0));

    console.log("Tensors in memory: " + tf?.memory().numTensors);
  };

  const gatherDataForClass = (classNumber: number) => {
    if (videoRef.current && videoPlaying) {
      let imageFeatures = tf?.tidy(function () {
        if (!tf || !mobileNet) {
          return;
        }

        let videoFrameAsTensor = tf.browser.fromPixels(videoRef.current);
        let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [224, 224],
          true
        );
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobileNet.predict(normalizedTensorFrame.expandDims()).squeeze();
      });

      setTrainingData([
        ...trainingData,
        { input: imageFeatures, output: classNumber },
      ]);

      // Capture the current frame and add it to the capturedImages array
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
      const capturedImage = canvas.toDataURL("image/png");
      setCapturedImages((prevImages) => {
        const classKey = `class_${classNumber + 1}`;
        return {
          ...prevImages,
          [classKey]: [capturedImage, ...prevImages[classKey]],
        };
      });
    }
  };

  return (
    <main className="flex min-h-screen flex-col gap-96 items-center justify-center p-12">
      <div className="flex flex-col gap-4">
        <Heading
          as="h1"
          size="7"
          weight="medium"
          className="text-center uppercase tracking-widest"
        >
          TensorFlow Transfer Learning
        </Heading>
        <Text as="p" size="3" className="text-center">
          A demo by{" "}
          <a
            href="https://davebitter.com"
            target="__blank"
            className="text-indigo-300 font-bold"
          >
            Dave Bitter
          </a>
        </Text>

        <div>
          <div className="flex flex-col items-stretch gap-4 lg:flex-row">
            <Card>
              <div className="flex flex-col gap-4">
                <video
                  className="flex rounded-lg"
                  ref={videoRef}
                  autoPlay
                  muted
                ></video>

                <div className="flex flex-col items-stretch gap-4 lg:flex-row lg:items-center">
                  <Text
                    as="label"
                    size="2"
                    className="flex flex-col-reverse lg:flex-row items-center h-full gap-4 lg:whitespace-nowrap"
                  >
                    <Switch
                      size="3"
                      checked={videoPlaying}
                      onClick={toggleWebcam}
                    />
                    Webcam
                  </Text>
                  {Array.from({ length: numberOfClasses }).map((_, i) => (
                    <Button
                      key={i}
                      onClick={() => gatherDataForClass(i)}
                      disabled={
                        !videoPlaying || isTraining || !readyToTrainAndPredict
                      }
                    >
                      {readyToTrainAndPredict ? <CameraIcon /> : <Spinner />}{" "}
                      Gather data class {i + 1}
                    </Button>
                  ))}
                  <Button
                    onClick={trainAndPredict}
                    disabled={
                      Object.values(capturedImages).every(
                        (images) => images.length === 0
                      ) || isTraining
                    }
                  >
                    {isTraining ? (
                      <>
                        {" "}
                        <Spinner /> Training
                      </>
                    ) : (
                      <>
                        <EyeOpenIcon />
                        Train & Predict
                      </>
                    )}
                  </Button>

                  <Button
                    onClick={reset}
                    disabled={
                      Object.values(capturedImages).every(
                        (images) => images.length === 0
                      ) || isTraining
                    }
                  >
                    <ResetIcon /> Reset
                  </Button>
                </div>
              </div>
            </Card>
            <div className="flex flex-col gap-4 w-[30vw]">
              {Array.from({ length: numberOfClasses }).map((_, i) => (
                <Card key={i} className="flex-1">
                  <div className="flex flex-col items-stretch justify-between h-full gap-4">
                    <Text as="h2" size="5" align="center">
                      Class {i + 1}
                    </Text>
                    <ul className="grid grid-cols-8 gap-2 rounded-md max-w-96 max-h-56 overflow-auto flex-1">
                      {capturedImages[`class_${i + 1}`].map((image, index) => (
                        <li key={index}>
                          <img src={image} alt="" />
                        </li>
                      ))}
                    </ul>
                    <progress
                      className="w-full [&::-webkit-progress-bar]:rounded-md [&::-webkit-progress-value]:rounded-md [&::-webkit-progress-bar]:bg-slate-200 [&::-webkit-progress-value]:bg-blue-400 [&::-moz-progress-bar]:bg-blue-400"
                      value={progressBarValues[i]}
                      max="1"
                    ></progress>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>
      <ToastContainer />
    </main>
  );
}
