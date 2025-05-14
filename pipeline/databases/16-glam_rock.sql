-- Script to list all Glam rock bands ranked by longevity (until 2020)
SELECT name AS band_name,
       IFNULL(split, 2020) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
