const { chromium } = require("playwright-chromium");
const { postHouses } = require("./api");

// Web scrapper for pisos.com
const scrapperLinks = async (startPage, search, city) => {
  const browser = await chromium.launch({
    headless: false,
    defaultViewport: null
  });
  const page = await browser.newPage();
  const url = `https://www.pisos.com/${search}/pisos-${city}/ultimasemana/${startPage}`;

  await page.goto(url, { waitUntil: "networkidle" });

  // Accept cookies
  await page.waitForSelector("#didomi-notice-agree-button");
  await page.click("#didomi-notice-agree-button");

  let [pages] = await page.$$(".grid__title");
  pages = await pages?.textContent();
  pages = pages?.replace(/[^0-9]/g, "");
  pages = Math.ceil(pages / 30);

  for (let i = startPage; i <= pages; i++) {
    const houses = await page.$$(".ad-preview");
    const links = [];

    for (let x = 0; x < houses.length; x++) {
      const link = await houses[x]?.getAttribute("data-lnk-href");
      links.push(link);
    }

    for (let j = 1; j <= links.length; j++) {
      let link = links[j];

      if (!link) continue;
      link = `https://www.pisos.com${link}`;

      await page.goto(link);

      let [characteristics] = await page.$$(".characteristics");
      characteristics = await characteristics?.textContent();
      characteristics = characteristics
        ?.replace(/\n\s*\n/g, "\n")
        ?.replace(/^[ \t]+|[ \t]+$/gm, "");

      if (!characteristics) continue;

      // Expresiones regulares para extraer la información relevante
      const regexBuiltArea = /Superficie construida\n:\s*(\d+)\s*m²/;
      const regexUsableArea = /Superficie útil\n:\s*(\d+)\s*m²/;
      const regexBedrooms = /Habitaciones\n:\s*(\d+)/;
      const regexBathrooms = /Baños\n:\s*(\d+)/;
      const regexFloor = /Planta\n:\s*(.+)/;
      const regexAge = /Antigüedad\n:\s*(.+)/;
      const regexCondition = /Conservación\n:\s*(.+)/;
      const regexGarage = /Garaje\n:\s*(.+)/;
      const regexOrientation = /Orientación\n:\s*(.+)/;
      const regexEnergyCertificate = /Clasificación:\n:\s*(.+)/;
      const regexSwimmingPool = /Piscina\s*:\s*([^\n]+)/;
      const regexCommunityExpenses = /Gastos de comunidad\s*:\s*([^\n]+)/;
      const regexAirConditioning = /Aire acondicionado\s*:\s*([^\n]+)/;
      const regexHouseHeating = /Calefacción\s*:\s*([^\n]+)/;
      const regexHouseType = /Tipo de casa\s*:\s*([^\n]+)/;

      // Extract data using regular expressions
      const builtArea = characteristics.match(regexBuiltArea)?.[1] || NaN;
      const usableArea = characteristics.match(regexUsableArea)?.[1];
      const bedrooms = characteristics.match(regexBedrooms)?.[1];
      const bathrooms = characteristics.match(regexBathrooms)?.[1];
      const floor = characteristics.match(regexFloor)?.[1];
      const age = characteristics.match(regexAge)?.[1];
      const condition = characteristics.match(regexCondition)?.[1];
      const garage = characteristics.match(regexGarage)?.[1];
      const orientation = characteristics.match(regexOrientation)?.[1];
      const energyCertificate = characteristics.match(
        regexEnergyCertificate
      )?.[1];
      const swimmingPoolType = characteristics.match(regexSwimmingPool)?.[1];
      const communityExpenses = characteristics.match(
        regexCommunityExpenses
      )?.[1];
      const airConditioning = characteristics.match(regexAirConditioning)?.[1];
      const houseHeatingType = characteristics.match(regexHouseHeating)?.[1];
      const houseType = characteristics.match(regexHouseType)?.[1];
      const elevator = characteristics.indexOf("Ascensor") > 0;
      const houseHeating = characteristics.indexOf("Calefacción") > 0;
      const terrace = characteristics.indexOf("Terraza") > 0;
      const swimmingPool = characteristics.indexOf("Piscina") > 0;

      let [info] = await page.$$(".maindata-info");
      info = await info?.textContent();
      const title = info?.split("\n")?.[2].trim();
      const location = info?.split("\n")?.[3].trim();

      let [price] = await page.$$(".priceBox-price");
      price = await price?.textContent();

      const isRent = search === "alquiler-residencial";

      const regexRent = /(\d+(?:\.\d+)?)\s*€\/mes/;
      const regexBuy = /\s*([\d.,]+)\s*€/;
      const result = isRent ? regexRent.exec(price) : regexBuy.exec(price);
      price = result?.[1];

      let [photo] = await page.$$(".mainphoto-image-print");

      photo = await photo?.getAttribute("src");

      const newHouse = [
        city,
        link,
        price,
        title,
        location,
        builtArea,
        usableArea,
        bedrooms,
        bathrooms,
        floor,
        age,
        condition,
        garage,
        orientation,
        energyCertificate,
        communityExpenses,
        airConditioning,
        elevator,
        houseHeating,
        houseHeatingType,
        terrace,
        swimmingPool,
        swimmingPoolType,
        houseType,
        photo
      ];

      const range = isRent ? "rent" : "buy";
      await postHouses([newHouse], range);

      console.log(j, i, "/", pages);
    }

    const url = `https://www.pisos.com/${search}/pisos-${city}/${i}`;

    await page.goto(url);
  }

  browser.close();
};

const asyncLoop = async () => {
  const cities = ["madrid", "malaga", "barcelona"];

  for (let x = 2; x < cities.length; x++) {
    await scrapperLinks(1, "venta", cities[x]);
  }
};

asyncLoop();
