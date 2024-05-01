import { useEffect, useRef } from "react";
import { showToast } from "../utils/showToast";

export const useTensorFlow = () => {
  const tf = useRef<any>(null);

  useEffect(() => {
    showToast("Loading TensorFlow");

    import("@tensorflow/tfjs").then((tfModule) => {
      tf.current = tfModule;
    });
  }, []);

  useEffect(() => {
    if (tf.current) {
      showToast("Loaded TensorFlow");
    }
  }, [tf.current]);

  return tf.current;
};
