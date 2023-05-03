import "./App.css";
import HouseCard from "./components/HouseCard";

import houses from "./houses_to_buy.json";

const App = () => {
  return (
    <div>
      <div className="houses-container">
        {houses?.slice(0, 100).map((item, index) => (
          <HouseCard {...item} key={index} />
        ))}
      </div>
    </div>
  );
};

export default App;
