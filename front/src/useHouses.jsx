import { useState } from "react";
import housesJson from "./houses_to_buy.json";
import { shouldBuyThisHouse } from "./utils";

const pagesLength = 20;
const housesFiltered = housesJson.filter((house) => house.link);
const pages = parseInt((housesFiltered.length / pagesLength).toFixed());

const useCharacters = () => {
  const [currentPage, setCurrentPage] = useState(1);
  let houses = housesFiltered
    .map((house) => ({
      ...house,
      ...shouldBuyThisHouse(house)
    }))
    .sort((b, a) => a.grossReturn - b.grossReturn);

  houses = houses.slice(
    (currentPage - 1) * pagesLength,
    currentPage * pagesLength
  );

  const handlePaginationClick = (e) => {
    setCurrentPage(parseInt(e.target.innerText));
  };

  const paginationButtons = () => {
    const delta = 1;
    const left = currentPage - delta;
    const right = currentPage + delta + 1;
    let result = [];

    result = Array.from({ length: pages }, (_, k) => k + 1).filter((i) => {
      if (i === 1 || i === pages) return true;
      return i && i >= left && i < right;
    });

    return result;
  };

  return {
    houses,
    currentPage,
    paginationButtons,
    handlePaginationClick
  };
};

export default useCharacters;
