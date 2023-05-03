const HouseCard = (house) => {
  const { photo, title, link, builtArea, price, rentPrice, cashOnCashReturn } =
    house;

  return (
    <div className="house">
      <img src={photo} alt={title} loading="lazy" />
      <a href={link}>
        {title}, {builtArea} m²
      </a>
      <p>Precio de compra: {price}</p>
      <p>Precio de alquiler sugerido: {rentPrice}</p>
      <p>Rentabilidad anual bruta: {cashOnCashReturn}</p>
    </div>
  );
};

export default HouseCard;
