import { useEffect, useState } from "react";
import { showToast } from "../utils/showToast";

export const useTensorFlow = () => {
  const [tf, setTf] = useState<any>(null);

  useEffect(() => {
    showToast("Loading TensorFlow");

    import("@tensorflow/tfjs").then((tfModule) => {
      setTf(tfModule);
    });
  }, []);

  useEffect(() => {
    if (tf) {
      showToast("Loaded TensorFlow");
    }
  }, [tf]);

  return tf;
};
