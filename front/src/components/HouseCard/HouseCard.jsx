import { BiBed, BiBath, BiHomeAlt2 } from "react-icons/bi";

import styles from "./HouseCard.module.scss";
const HouseCard = ({
  photo,
  title,
  builtArea,
  price,
  rentPrice,
  cashOnCashReturn,
  bedrooms = "1",
  bathrooms = "1"
}) => {
  const [titleFirstWord, ...restTitle] = title.split(" ");

  const priceColor =
    cashOnCashReturn > 5
      ? styles.HouseCard__body__rentPrice__green
      : styles.HouseCard__body__rentPrice__orange;

  return (
    <div className={styles.HouseCard}>
      <img
        className={styles.HouseCard__image}
        src={photo}
        alt={title}
        loading="lazy"
      />
      <div className={styles.HouseCard__body}>
        <h3>
          <span className={styles.HouseCard__body__price}>{price}€</span>
          <span className={styles.HouseCard__body__title}>
            <strong>{titleFirstWord}</strong> {restTitle.join(" ")}
          </span>
        </h3>

        <ul className={styles.HouseCard__body__features}>
          <li className={styles.HouseCard__body__features__item}>
            {bedrooms} {bathrooms > 1 ? "habs" : "hab"} <BiBed />
          </li>
          <li className={styles.HouseCard__body__features__item}>
            {bathrooms} {bathrooms > 1 ? "baños" : "baño"} <BiBath />{" "}
          </li>
          <li className={styles.HouseCard__body__features__item}>
            {builtArea} m²
            <BiHomeAlt2 />
          </li>
        </ul>
        <p className={styles.HouseCard__body__rentPrice}>
          Precio de alquiler sugerido:
          <span className={styles.HouseCard__body__rentPrice__green}>
            {" "}
            {rentPrice.toFixed()}€
          </span>
        </p>

        <p className={styles.HouseCard__body__rentPrice}>
          Rentabilidad anual bruta:
          <span className={priceColor}> {cashOnCashReturn.toFixed(2)}%</span>
        </p>
      </div>
    </div>
  );
};

export default HouseCard;
