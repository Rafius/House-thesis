import HouseCard from "./components/HouseCard";
import housesJson from "./houses_to_buy.json";
import { shouldBuyThisHouse } from "./utils";
import styles from "./App.module.scss";

const App = () => {
  const houses = housesJson
    .map((house) => ({
      ...house,
      ...shouldBuyThisHouse(house)
    }))
    .filter((house) => house.isHouseInteresting)
    .sort((b, a) => a.cashOnCashReturn - b.cashOnCashReturn);

  return (
    <div className={styles.HousesContainer}>
      {houses.slice(0, 10).map((house) => (
        <HouseCard {...house} key={house.link} {...shouldBuyThisHouse(house)} />
      ))}
    </div>
  );
};

export default App;
