import "./App.css";
import HouseCard from "./components/HouseCard";
import housesJson from "./houses_to_buy.json";
import { shouldBuyThisHouse } from "./utils";

const App = () => {
  const houses = housesJson
    .map((house) => ({
      ...house,
      ...shouldBuyThisHouse(house)
    }))
    .filter((house) => house.isHouseInteresting)
    .sort((b, a) => a.cashOnCashReturn - b.cashOnCashReturn);

  return (
    <div>
      <div className="houses-container">
        {houses.map((house, index) => (
          <HouseCard {...house} key={index} {...shouldBuyThisHouse(house)} />
        ))}
      </div>
    </div>
  );
};

export default App;
