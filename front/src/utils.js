export const shouldBuyThisHouse = ({ price, rentPrice }) => {
  const housePrice = parseInt(price) * 1000;
  const yearlyRentPrice = parseInt(rentPrice) * 12;
  const notary = 800;
  const vat = housePrice / 10;

  const grossReturn = (yearlyRentPrice / (housePrice + notary + vat)) * 100;

  return { grossReturn };
};
