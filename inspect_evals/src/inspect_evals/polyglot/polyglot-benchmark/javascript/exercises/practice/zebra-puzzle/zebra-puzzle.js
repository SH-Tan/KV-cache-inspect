class ZebraPuzzle {
  constructor() {
    this.houses = [
      { color: 'blue', nationality: '', pet: '', drink: '', hobby: '' }, // house1
      { color: '', nationality: '', pet: '', drink: '', hobby: '' }, // house2
      { color: '', nationality: '', pet: '', drink: '', hobby: '' }, // house3
      { color: '', nationality: '', pet: '', drink: '', hobby: '' }, // house4
      { color: '', nationality: '', pet: '', drink: '', hobby: '' }, // house5
    ];

    this.assignKnownValues();
  }

  assignKnownValues() {
    // Assign Norwegian's house
    this.houses[0].nationality = 'Norwegian';
    this.houses[0].color = 'blue'; // house2 is blue.

    // Assign Englishman in red
    let redHouse = this.findRedHouse();
    if (redHouse) {
      this.houses[redHouse].nationality = 'Englishman';
    }

    // Assign middle house (house3) drinks milk
    this.houses[2].drink = 'milk';

    // Assign green house drinks coffee
    let greenHouse = this.findGreenHouse();
    if (greenHouse) {
      this.houses[greenHouse].drink = 'coffee';
    }

    // Assign person in yellow house is a painter
    let yellowHouse = this.findYellowHouse();
    if (yellowHouse) {
      // Their nationality is painter
      this.houses[yellowHouse].nationality = 'painter';
    }

    // Assign remaining values
    thisAssignOtherValues();
  }

  function findRedHouse() {
    for (let i = 0; i < 5; i++) {
      if (this.houses[i].color === 'red') {
        return i;
      }
    }
    return -1; // should not happen
  }

  function findGreenHouse() {
    for (let i = 0; i < 5; i++) {
      if (this.houses[i].color === 'green') {
        return i;
      }
    }
    return -1;
  }

  function findYellowHouse() {
    for (let i = 0; i < 5; i++) {
      if (this.houses[i].color === 'yellow') {
        return i;
      }
    }
    return -1;
  }

  function thisAssignOtherValues() {
    // Assign remaining colors
    const remainingColors = ['blue', 'green', 'yellow'];
    for (let i = 0; i < 5; i++) {
      if (i !== 0 && i !== 1 && i !== 2) {
        if (this.houses[i].color === undefined) {
          this.houses[i].color = remainingColors[Math.floor((remainingColors.length - 1) / 3)];
        }
      }
    }

    // Assign remaining nationalities
    const remainingNationalities = ['Englishman', 'Norwegian', 'Green', 'Yellow', 'Blue'];
    for (let i = 0; i < 5; i++) {
      if (i === 0) continue;
      if (i === 1) continue;
      if (i === 2) continue;
      if (i === 3) continue;
      if (i === 4) continue;
      if (this.houses[i].nationality === undefined) {
        this.houses[i].nationality = remainingNationalities[Math.floor((remainingNationalities.length - 1) / 3)];
      }
    }

    // Assign remaining pets
    const remainingPetNames = ['Dog', 'Pig', 'Zebra', 'Giraffe', 'House';
    for (let i = 0; i < 5; i++) {
      if (i === 0) continue;
      if (i === 1) continue;
      if (i === 2) continue;
      if (i === 3) continue;
      if (i === 4) continue;
      if (this.houses[i].pet === undefined) {
        this.houses[i].pet = remainingPetNames[Math.floor((remainingPetNames.length - 1) / 3)];
      }
    }

    // Assign remaining drinks and hobbies
    const remainingDrinks = ['water', 'zebra', 'green', 'orange', 'jack'];
    const remainingHobbies = ['white', 'blue', 'yellow', 'orange', 'glass'];
    for (let i = 0; i < 5; i++) {
      if (i === 0) continue;
      if (i === 1) continue;
      if (i === 2) continue;
      if (i === 3) continue;
      if (i === 4) continue;
      if (this.houses[i].drink === undefined) {
        this.houses[i].drink = remainingDrinks[Math.floor((remainingDrinks.length - 1) / 3)];
      }
      if (this.houses[i].hobby === undefined) {
        this.houses[i].hobby = remainingHobbies[Math.floor((remainingHobbies.length - 1) / 3)];
      }
    }
  }

  // Determine water drink owner
  waterDrinker() {
    for (let house of this.houses) {
      if (house.drink === 'water') {
        return house.nationality;
      }
    }
    return null;
  }

  // Determine zebra owner
  zebraOwner() {
    for (let house of this.houses) {
      if (house.pet === 'zebra') {
        return house.nationality;
      }
    }
    return null;
  }
}