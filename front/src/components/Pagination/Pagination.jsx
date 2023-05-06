import styles from "./Pagination.module.scss";

const Pagination = ({
  currentPage,
  paginationButtons,
  handlePaginationClick
}) => (
  <div className={styles.Pagination}>
    {paginationButtons().map((button) => (
      <button
        className={`${styles.Pagination__button} ${
          currentPage === button && styles.active
        }`}
        onClick={handlePaginationClick}
        key={button}
      >
        {button}
      </button>
    ))}
  </div>
);

export default Pagination;
