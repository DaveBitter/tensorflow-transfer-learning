"use client";
import { CameraIcon, EyeOpenIcon, ResetIcon } from "@radix-ui/react-icons";
import { Button, Card, Heading, Switch, Text, Spinner } from "@radix-ui/themes";
import { useEffect, useRef, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const toastOptions = {
  position: "bottom-right",
  theme: "dark",
  hideProgressBar: true,
};

let gatherDataState = -1;
export default function Home() {
  const [amountOfClasses] = useState(2);
  const [trainingDataInputs, setTrainingDataInputs] = useState([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState([]);
  const mobilenet = useRef(null);
  const predict = useRef(null);
  const model = useRef(null);
  const [tf, setTf] = useState(null);

  useEffect(() => {
    toast("Loading TensorFlow", toastOptions);

    import("@tensorflow/tfjs").then((tfModule: any) => {
      setTf(tfModule);
    });
  }, []);

  useEffect(() => {
    if (tf) {
      toast("Loaded TensorFlow", toastOptions);
    }
  }, [tf]);

  const videoRef = useRef<HTMLVideoElement>(null);

  const [readyToGatherData, setReadyToGatherData] = useState(false);
  useEffect(() => {
    if (!tf) {
      return () => {};
    }

    async function loadMobileNetFeatureModel() {
      const URL =
        "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

      mobilenet.current = await tf.loadGraphModel(URL, { fromTFHub: true });

      // Warm up the model by passing zeros through it once.
      tf.tidy(function () {
        let answer = mobilenet.current.predict(tf.zeros([1, 224, 224, 3]));

        setReadyToGatherData(true);
      });
    }

    // Call the function immediately to start loading.
    toast("Loading MobileNet", toastOptions);
    loadMobileNetFeatureModel().then(() => {
      model.current = tf.sequential();
      model.current.add(
        tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
      );
      model.current.add(
        tf.layers.dense({ units: amountOfClasses, activation: "softmax" })
      );

      model.current.summary();

      // Compile the model with the defined optimizer and specify a loss function to use.
      model.current.compile({
        // Adam changes the learning rate over time which is useful.
        optimizer: "adam",
        // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
        // Else categoricalCrossentropy is used if more than 2 classes.
        loss:
          amountOfClasses === 2
            ? "binaryCrossentropy"
            : "categoricalCrossentropy",
        // As this is a classification problem you can record accuracy in the logs too!
        metrics: ["accuracy"],
      });
      toast("Loaded MobileNet", toastOptions);
    });
  }, [tf]);

  const [videoPlaying, setVideoPlaying] = useState(false);
  const toggleWebcam = async () => {
    if (videoPlaying) {
      videoRef.current.srcObject.getVideoTracks().forEach((track) => {
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
      .query({ name: "camera" })
      .then(function (permissionStatus) {
        if (permissionStatus.state === "granted") {
          toggleWebcam();
        }
      });
  }, []);

  const [progressBar1Value, setProgressBar1Value] = useState(0);
  const [progressBar2Value, setProgressBar2Value] = useState(0);

  function predictLoop() {
    if (predict.current) {
      tf.tidy(function () {
        let videoFrameAsTensor = tf.browser
          .fromPixels(videoRef.current)
          .div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [224, 224],
          true
        );

        let imageFeatures = mobilenet.current.predict(
          resizedTensorFrame.expandDims()
        );
        let prediction = model.current.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();

        setProgressBar1Value(predictionArray[0]);
        setProgressBar2Value(predictionArray[1]);

        console.log(predictionArray);
      });

      window.requestAnimationFrame(predictLoop);
    }
  }

  const [isTraining, setIsTraining] = useState(false);
  const trainAndPredict = async () => {
    setIsTraining(true);
    toast("Started training", toastOptions);
    predict.current = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
    let oneHotOutputs = tf.oneHot(outputsAsTensor, amountOfClasses);
    let inputsAsTensor = tf.stack(trainingDataInputs);

    await model.current.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true,
      batchSize: 5,
      epochs: 10,
      callbacks: { onEpochEnd: logProgress },
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    predict.current = true;
    predictLoop();
    toast("Finished training", toastOptions);
    setIsTraining(false);
  };

  function logProgress(epoch, logs) {
    console.log("Data for epoch " + epoch, logs);
  }

  const [capturedImages, setCapturedImages] = useState({
    class1: [],
    class2: [],
  });
  function dataGatherLoop() {
    if (videoPlaying && gatherDataState !== -1) {
      let imageFeatures = tf.tidy(function () {
        let videoFrameAsTensor = tf.browser.fromPixels(videoRef.current);
        let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [224, 224],
          true
        );
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenet.current
          .predict(normalizedTensorFrame.expandDims())
          .squeeze();
      });

      setTrainingDataInputs([...trainingDataInputs, imageFeatures]);
      setTrainingDataOutputs([...trainingDataOutputs, gatherDataState]);

      // Capture the current frame and add it to the capturedImages array
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
      const capturedImage = canvas.toDataURL("image/png");
      setCapturedImages((prevImages) => {
        const classKey = gatherDataState === 0 ? "class1" : "class2";
        return {
          ...prevImages,
          [classKey]: [capturedImage, ...prevImages[classKey]],
        };
      });

      setTimeout(() => {
        window.requestAnimationFrame(dataGatherLoop);
      }, 500);
    }
  }

  const gatherDataForClass = (e) => {
    let classNumber = parseInt(e.target.getAttribute("data-1hot"));
    gatherDataState = gatherDataState === -1 ? classNumber : -1;
    dataGatherLoop();
  };

  const stopGatherDataForClass = (e) => {
    gatherDataState = -1;
  };

  const reset = () => {
    predict.current = null;

    setTrainingDataInputs([]);
    setTrainingDataOutputs([]);
    setCapturedImages({ class1: [], class2: [] });
    setProgressBar1Value(0);
    setProgressBar2Value(0);

    console.log("Tensors in memory: " + tf.memory().numTensors);
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
                  <Button
                    onMouseDown={gatherDataForClass}
                    onMouseUp={stopGatherDataForClass}
                    onMouseLeave={stopGatherDataForClass}
                    data-1hot="0"
                    disabled={!videoPlaying || isTraining || !readyToGatherData}
                  >
                    {readyToGatherData ? <CameraIcon /> : <Spinner />} Gather
                    data class 1
                  </Button>
                  <Button
                    onMouseDown={gatherDataForClass}
                    onMouseUp={stopGatherDataForClass}
                    data-1hot="1"
                    disabled={!videoPlaying || isTraining || !readyToGatherData}
                  >
                    {readyToGatherData ? <CameraIcon /> : <Spinner />}
                    Gather data class 2
                  </Button>
                  <Button
                    onClick={trainAndPredict}
                    disabled={
                      (!capturedImages.class1.length &&
                        !capturedImages.class2.length) ||
                      isTraining
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
                      (!capturedImages.class1.length &&
                        !capturedImages.class2.length) ||
                      isTraining
                    }
                  >
                    <ResetIcon /> Reset
                  </Button>
                </div>
              </div>
            </Card>
            <div className="flex flex-col gap-4 w-[30vw]">
              <Card className="flex-1">
                <div className="flex flex-col items-stretch justify-between h-full gap-4">
                  <Text as="h2" size="5" align="center">
                    Class 1
                  </Text>
                  <ul className="grid grid-cols-8 gap-2 rounded-md max-w-96 max-h-56 overflow-auto flex-1">
                    {capturedImages.class1.map((image, index) => (
                      <li key={index}>
                        <img src={image} alt="" />
                      </li>
                    ))}
                  </ul>
                  <progress
                    className="w-full [&::-webkit-progress-bar]:rounded-md [&::-webkit-progress-value]:rounded-md [&::-webkit-progress-bar]:bg-slate-200 [&::-webkit-progress-value]:bg-blue-400 [&::-moz-progress-bar]:bg-blue-400"
                    value={progressBar1Value}
                    max="1"
                  ></progress>
                </div>
              </Card>
              <Card className="flex-1">
                <div className="flex flex-col items-stretch justify-between h-full gap-4">
                  <Text as="h2" size="5" align="center">
                    Class 2
                  </Text>
                  <ul className="grid grid-cols-8 gap-2 rounded-md max-w-96 max-h-56 overflow-auto flex-1">
                    {capturedImages.class2.map((image, index) => (
                      <li key={index}>
                        <img src={image} alt="" />
                      </li>
                    ))}
                  </ul>
                  <progress
                    className="w-full [&::-webkit-progress-bar]:rounded-md [&::-webkit-progress-value]:rounded-md [&::-webkit-progress-bar]:bg-slate-200 [&::-webkit-progress-value]:bg-orange-400 [&::-moz-progress-bar]:bg-orange-400"
                    value={progressBar2Value}
                    max="1"
                  ></progress>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
      <ToastContainer />
    </main>
  );
}
