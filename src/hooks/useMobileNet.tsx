import { useRef, useEffect, useState } from "react";
import { showToast } from "../utils/showToast";

export const useMobileNet = ({
  tf,
  numberOfClasses,
}: {
  tf: any;
  numberOfClasses: number;
}) => {
  const model = useRef<any>(null);
  const mobileNet = useRef<any>(null);
  const [readyToTrainAndPredict, setReadyToTrainAndPredict] = useState(false);

  useEffect(() => {
    if (!tf) {
      return () => {};
    }

    async function loadMobileNetFeatureModel() {
      if (!tf) {
        return;
      }

      const URL =
        "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

      mobileNet.current = await tf.loadGraphModel(URL, {
        fromTFHub: true,
      });

      // Warm up the model by passing zeros through it once.
      tf.tidy(function () {
        mobileNet.current?.predict(tf.zeros([1, 224, 224, 3]));
      });
    }

    // Call the function immediately to start loading.
    showToast("Loading MobileNet");
    loadMobileNetFeatureModel().then(() => {
      if (!tf) {
        return;
      }

      model.current = tf.sequential();

      if (!model.current) {
        return;
      }

      model.current.add(
        tf.layers.dense({
          inputShape: [1024],
          units: 128,
          activation: "relu",
        })
      );
      model.current.add(
        tf.layers.dense({
          units: numberOfClasses,
          activation: "softmax",
        })
      );

      model.current.summary();

      // Compile the model with the defined optimizer and specify a loss function to use.
      model.current.compile({
        // Adam changes the learning rate over time which is useful.
        optimizer: "adam",
        // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
        // Else categoricalCrossentropy is used if more than 2 classes.
        loss:
          numberOfClasses === 2
            ? "binaryCrossentropy"
            : "categoricalCrossentropy",
        // As this is a classification problem you can record accuracy in the logs too!
        metrics: ["accuracy"],
      });
      setReadyToTrainAndPredict(true);
      showToast("Loaded MobileNet");
    });
  }, [tf]);

  return {
    model: model.current,
    mobileNet: mobileNet.current,
    readyToTrainAndPredict,
  };
};
