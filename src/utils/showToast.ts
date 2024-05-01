import { ToastOptions, toast } from "react-toastify";

const toastOptions: ToastOptions = {
  position: "bottom-right",
  theme: "dark",
  hideProgressBar: true,
};

export const showToast = (message: string) => {
  toast(message, toastOptions);
};
