-- 15-fans.sql

-- Afficher le nombre total de fans par pays d'origine (origin)
-- Ordonné par nombre de fans décroissant

SELECT
    origin,
    SUM(nb_fans) AS nb_fans
FROM
    metal_bands
GROUP BY
    origin
ORDER BY
    nb_fans DESC;
