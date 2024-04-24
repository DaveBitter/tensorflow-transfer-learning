export const startViewTransition = (cb: any) => {
  if (!document.startViewTransition) {
    cb();
  } else {
    document.startViewTransition(() => {
      cb();
    });
  }
};
