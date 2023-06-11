const calculateMortgageCost = (housePrice) => {
  const years = 30;
  const months = years * 12;
  const mortgageTax = 3.5 / 100 / 12;
  const loan = housePrice * 0.8;

  const monthlyPayment =
    (loan * (mortgageTax * Math.pow(1 + mortgageTax, months))) /
    (Math.pow(1 + mortgageTax, months) - 1);

  const mortgageCost = monthlyPayment * months - loan;

  return mortgageCost / years;
};

export const shouldBuyThisHouse = (house) => {
  let housePrice = house.price;
  housePrice = parseInt(housePrice) * 1000;
  const yearlyRentPrice = parseInt(house.rentPrice) * 12;
  const deposit = housePrice / 5;
  const notary = 800;
  const vat = housePrice / 10;
  const unpaidInsurance = 230;
  const garbageTax = 100;
  const homeInsurance = 200;
  const lifeInsurance = 100;
  const communityFees = 40 * 12;
  const propertyTaxes = housePrice / 100 / 12;
  const maintenance = yearlyRentPrice / 20;
  const vacancyPeriods = yearlyRentPrice / 20;
  const mortgageCost = calculateMortgageCost(housePrice);

  const yearEarnings =
    yearlyRentPrice -
    unpaidInsurance -
    garbageTax -
    homeInsurance -
    lifeInsurance -
    communityFees -
    propertyTaxes -
    maintenance -
    vacancyPeriods -
    mortgageCost;

  const cashFlow = yearEarnings - (housePrice - deposit) / 30;
  const cashOnCashReturn = (cashFlow / (deposit + notary + vat)) * 100;

  const grossReturn = (yearlyRentPrice / (housePrice + notary + vat)) * 100;
  const isHouseInteresting = cashOnCashReturn > 8;

  return { cashOnCashReturn, isHouseInteresting, grossReturn };
};
