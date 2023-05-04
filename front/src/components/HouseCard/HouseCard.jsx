import styles from "./HouseCard.module.scss";
const HouseCard = ({
  photo,
  title,
  link,
  builtArea,
  price,
  rentPrice,
  cashOnCashReturn,
  bedrooms = "1",
  bathrooms = "1"
}) => {
  console.log(bedrooms);
  return (
    <div className={styles.HouseCard}>
      <img
        className={styles.HouseCard__image}
        src={photo}
        alt={title}
        loading="lazy"
      />
      <p> {price}</p>
      <p className={styles.HouseCard__title}> {title}</p>
      <ul className={styles.HouseCard__features}>
        <li>{bedrooms}</li>
        <li>{bathrooms}</li>
        <li>{builtArea}</li>
      </ul>
      {/* <p>Precio de alquiler sugerido: {rentPrice}</p>
      <p>Rentabilidad anual bruta: {cashOnCashReturn}</p> */}
    </div>
  );
};

export default HouseCard;
