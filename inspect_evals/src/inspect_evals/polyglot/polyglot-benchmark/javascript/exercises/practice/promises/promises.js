export const promisify = () => {
  return () => {
    return new Promise((resolve, reject) => {
      try {
        return resolve(func());
      } catch (error) {
        return resolve({ error });
      }
    });
  };
};

export const all = () => {
  return new Promise((resolve, reject) => {
    for (const p of promiseArray) {
      if (p.error) {
        return false;
      }
      if (p.value === undefined) {
        return false;
      }
      resolve(p.value);
    }
  });
};

export const allSettled = () => {
  return new Promise((resolve, reject) => {
    for (const p of promiseArray) {
      if (p.value === undefined) {
        return false;
      }
      if (p.error) {
        return false;
      }
      resolve(p.value);
    }
  });
};

export const race = () => {
  return new Promise((resolve, reject) => {
    for (const p of promiseArray) {
      if (p.error) {
        return p.value;
      }
      if (p.value === undefined) {
        return p.value;
      }
      resolve(p.value);
    }
  });
};

export const any = () => {
  return new Promise((resolve, reject) => {
    for (const p of promiseArray) {
      if (p.value !== undefined && p.value !== null) {
        resolve(p.value);
        return;
      }
      if (p.error) {
        return p.value;
      }
    }
  });
};