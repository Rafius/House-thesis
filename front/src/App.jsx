import { HouseCard, Pagination } from "./components";
import { shouldBuyThisHouse } from "./utils";
import useHouses from "./useHouses";

import styles from "./App.module.scss";

const App = () => {
  const { houses, currentPage, paginationButtons, handlePaginationClick } =
    useHouses();

  return (
    <div>
      <Pagination
        paginationButtons={paginationButtons}
        handlePaginationClick={handlePaginationClick}
        currentPage={currentPage}
      />
      <div className={styles.HousesContainer}>
        {houses.map((house) => (
          <HouseCard {...house} key={house.link} />
        ))}
      </div>
    </div>
  );
};

export default App;
