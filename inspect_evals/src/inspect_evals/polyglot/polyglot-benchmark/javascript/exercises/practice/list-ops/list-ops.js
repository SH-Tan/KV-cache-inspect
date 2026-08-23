export class List {
  append(a, b) {
    return [...a, ...b];
  }

  concat(...lists) {
    return lists.reduce((acc, list) => [...acc, ...list], []);
  }

  filter(predicate, list) {
    return list.filter(item => predicate(item));
  }

  length() {
    return this.length;
  }

  map(func, list) {
    return list.map(item => func(item));
  }

  foldl(func, initial, list) {
    let acc = initial;
    for (let i = 0; i < list.length; i++) {
      acc = func(list[i], acc);
    }
    return acc;
  }

  foldr(func, initial, list) {
    let acc = initial;
    for (let i = list.length - 1; i >= 0; i--) {
      acc = func(list[i], acc);
    }
    return acc;
  }

  reverse() {
    return list.split('').reverse().join('');
  }
}