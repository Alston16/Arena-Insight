few_shots = [
    {
        'input':"Find the total number of medals won by France in Olympic Games",
        'query': """SELECT count(ap.medal)
               FROM (
               SELECT event_id, MIN(part_id) AS first_part_id
               FROM athlete_part
               GROUP BY event_id,medal
               ) AS first_occurrence
               JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
               JOIN events ev ON ap.event_id = ev.event_id
               JOIN sports sp ON ev.sport_id = sp.sport_id
               JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
               WHERE cp.as_country LIKE "%France%" AND ap.medal IS NOT NULL;
          """
    },
    {
        'input' : "Retrieve the name of the athlete who won the most gold medals",
     'query' : """SELECT a.name
          FROM athletes a
          JOIN athlete_medals am ON a.athlete_id = am.athlete_id
          ORDER BY am.gold DESC
          LIMIT 1;"""
     },
     
    {
        'input': "Find the country with the highest number of Olympic participants",
        'query':"""SELECT cd.country_name
               FROM country_details cd
               JOIN country_participations cp ON cd.country_id = cp.country_id
               GROUP BY cd.country_name
               ORDER BY SUM(cp.total_count) DESC
               LIMIT 1;"""
     },
    {
        'input': "How many medals have Usain Bolt has won in Olympics" ,
        'query' : """select total from athletes a, athlete_medals b where a.athlete_id=b.athlete_id and a.name like "%Usain Bolt%";"""
     } ,
     {
         'input' : """How many gold medals has Michael Phelps won in Olympics""" ,
         'query': """select gold from athletes a, athlete_medals b where a.athlete_id=b.athlete_id and a.name like "%Michael Phelps%";"""
     },
    {
        'input': "Find the top 10 athletes with the their participations in most Olympic editions",
        'query' : """SELECT a.name
               FROM athletes a
               JOIN (
               SELECT athlete_id, COUNT(DISTINCT edition) AS participation_count
               FROM athlete_results
               GROUP BY athlete_id
               ) op ON a.athlete_id = op.athlete_id
               ORDER BY op.participation_count DESC
               LIMIT 10;"""
    },
    {
        'input' : "Find the most recent edition an athlete from Afghanistan won a medal:",
        'query' : """SELECT MAX(r.edition)
               FROM athletes a
               JOIN athlete_results r ON a.athlete_id = r.athlete_id
               WHERE r.medal_type IS NOT NULL AND a.noc = 'Afghanistan';
               """
     },
     {
         'input':"Name all the atheletes from India who were born after 1st January, 1990",
         'query': """SELECT name FROM athletes WHERE STR_TO_DATE(SUBSTRING_INDEX(born, ' in ', 1),'%d %M %Y') > '1990-01-01' and noc like '%India%';"""
     },
     {
         'input' : "How many medals has Australia men hockey team won",
         'query' : """SELECT COUNT(DISTINCT e.event_id) AS total_medals
                    FROM athlete_part ap
                    JOIN events e ON ap.event_id = e.event_id
                    JOIN sports s ON e.sport_id = s.sport_id
                    JOIN country_participations cp ON s.country_participation_id = cp.participation_id
                    JOIN country_details cd ON cp.country_id = cd.country_id
                    WHERE s.sport_name = 'Hockey'
                    AND e.event_name LIKE '%Men%'
                    AND cd.country_name LIKE '%Australia%'
                    AND ap.medal IS NOT NULL;
                    """
     },
     {
         'input':"Total number of medals won by France in Olympics",
         'query': """SELECT count(ap.medal)
               FROM (
               SELECT event_id, MIN(part_id) AS first_part_id
               FROM athlete_part
               GROUP BY event_id,medal
               ) AS first_occurrence
               JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
               JOIN events ev ON ap.event_id = ev.event_id
               JOIN sports sp ON ev.sport_id = sp.sport_id
               JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
               WHERE cp.as_country LIKE "%France%" AND ap.medal IS NOT NULL;
               """
     },
     {
        'input': "How many female athletes participated in 2020 Olympics from France",
        'query': "SELECT women_count FROM country_participations WHERE as_country LIKE '%France%' and edition = '2020 Summer Olympics';"
    },
    {
        'input': "How many gold, silver and bronze medals has India won in Olympics",
        'query': """SELECT 
                         SUM(CASE WHEN ap.medal = 'Gold' THEN 1 ELSE 0 END) AS gold_count,
                         SUM(CASE WHEN ap.medal = 'Silver' THEN 1 ELSE 0 END) AS silver_count,
                         SUM(CASE WHEN ap.medal = 'Bronze' THEN 1 ELSE 0 END) AS bronze_count
                         FROM (
                         SELECT event_id, MIN(part_id) AS first_part_id
                         FROM athlete_part
                         GROUP BY event_id, medal
                         ) AS first_occurrence
                         JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
                         JOIN events ev ON ap.event_id = ev.event_id
                         JOIN sports sp ON ev.sport_id = sp.sport_id
                         JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
                         WHERE cp.as_country LIKE "%India%" 
                         AND ap.medal IS NOT NULL;"""
    },
    {
        'input': "Get the edition of the Olympics with the highest number of participating countries",
        'query': """SELECT cp.edition
                         FROM country_participations cp
                         GROUP BY cp.edition
                         ORDER BY COUNT(cp.country_id) DESC
                         LIMIT 1;
                    """
    },
    {
        'input': "Get the top 5 athletes in Swimming sports",
        'query': """SELECT ap.name FROM athlete_part ap
                         JOIN events e ON ap.event_id = e.event_id
                         JOIN sports s ON e.sport_id = s.sport_id
                         WHERE s.sport_name like '%Swimming%'
                         AND ap.medal IS NOT NULL
                         GROUP BY ap.name
                         ORDER BY count(ap.medal) DESC
                         LIMIT 5;
                    """
    },
    {
        'input': "Get the top 5 athletes in Javelin event of Athletics sports",
        'query': """SELECT ap.name
                         FROM athlete_part ap
                         JOIN events e ON ap.event_id = e.event_id
                         JOIN sports s ON e.sport_id = s.sport_id
                         WHERE s.sport_name like '%Athletics%' 
                         AND e.event_name like '%Javelin%'
                         AND ap.medal IS NOT NULL
                         GROUP BY ap.name
                         ORDER BY COUNT(ap.medal) DESC
                         LIMIT 5;                    
                    """
    },
    {
        'input': "How many medals did Australia win in Athletics in 2016 Summer Olympics",
        'query': """SELECT count(ap.medal)
               FROM (
               SELECT event_id, MIN(part_id) AS first_part_id
               FROM athlete_part
               GROUP BY event_id,medal
               ) AS first_occurrence
               JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
               JOIN events ev ON ap.event_id = ev.event_id
               JOIN sports sp ON ev.sport_id = sp.sport_id
               JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
               WHERE sp.sport_name like '%Athletics%' AND cp.as_country LIKE "%Australia%" AND cp.edition = '2016 Summer Olympics' AND ap.medal IS NOT NULL;
               """
    },
    {
        'input': "Name the members of the mens hockey team who won Gold medal in 1948 Summer Olympics",
        'query': """SELECT ap.name
                         FROM athlete_part ap
                         JOIN events e ON ap.event_id = e.event_id
                         JOIN sports s ON e.sport_id = s.sport_id
                         JOIN country_participations cp ON s.country_participation_id = cp.participation_id
                         WHERE cp.edition = '1948 Summer Olympics'
                         AND s.sport_name = 'Hockey'
                         AND (e.event_name LIKE '% Men%' 
                              OR e.event_name LIKE 'Men %'
                              OR e.event_name LIKE '% Men %'
                              OR e.event_name = 'Men')
                         AND ap.medal = 'Gold';
                    """
    },
    {
        'input': "Which country won Silver medal in Womens Hockey at 2020 Summer Olympics",
        'query': """
                         SELECT DISTINCT cd.country_name
                         FROM country_details cd
                         JOIN country_participations cp ON cd.country_id = cp.country_id
                         JOIN sports s ON cp.participation_id = s.country_participation_id
                         JOIN events e ON s.sport_id = e.sport_id
                         JOIN athlete_part ap ON e.event_id = ap.event_id
                         WHERE s.sport_name = 'Hockey'  
                         AND cp.edition = '2020 Summer Olympics'   
                         AND (e.event_name LIKE '% Women%' 
                              OR e.event_name LIKE 'Women %'
                              OR e.event_name LIKE '% Women %'
                              OR e.event_name = 'Women')     
                         AND ap.medal = "Silver";
                    """
    },
    {
        'input': "What medals did Australia win in Mens Hockey in the last 10 editions",
        'query':"""
                        SELECT cp.edition, ap.medal
                        FROM (
                            SELECT event_id, MIN(part_id) AS first_part_id
                            FROM athlete_part
                            GROUP BY event_id,medal
                        ) AS first_occurrence
                        JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
                        JOIN events ev ON ap.event_id = ev.event_id
                        JOIN sports sp ON ev.sport_id = sp.sport_id
                        JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
                        JOIN (
                            SELECT DISTINCT edition
                            FROM country_participations
                            WHERE edition like "%Summer%"
                            ORDER BY edition DESC
                            LIMIT 10
                        ) latest_editions ON cp.edition = latest_editions.edition 
                        WHERE cp.as_country LIKE "%Australia%" 
                        AND sp.sport_name = 'Hockey'
                        AND (ev.event_name LIKE '% Men%' 
                            OR ev.event_name LIKE 'Men %'
                            OR ev.event_name LIKE '% Men %'
                            OR ev.event_name = 'Men')
                        AND ap.medal IS NOT NULL;
                   """
    },
    {
        'input': "What were the position of Indonesia athletes in Badminton in the last 2 editions of Olympics",
        'query': """SELECT cp.edition, ev.event_name, ap.position
                        FROM (
                            SELECT event_id, MIN(part_id) AS first_part_id
                            FROM athlete_part
                            GROUP BY event_id,medal
                        ) AS first_occurrence
                        JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
                        JOIN events ev ON ap.event_id = ev.event_id
                        JOIN sports sp ON ev.sport_id = sp.sport_id
                        JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
                        JOIN (
                            SELECT DISTINCT edition
                            FROM country_participations
                            WHERE edition like "%Summer%"
                            ORDER BY edition DESC
                            LIMIT 2
                        ) latest_editions ON cp.edition = latest_editions.edition 
                        WHERE cp.as_country LIKE "%Indonesia%" 
                        AND sp.sport_name = 'Badminton';
                    """
    },
    {
        'input': 'List top 5 countries with most medals in Olympics till 2020',
        'query': """SELECT cd.country_name, count(ap.medal) AS total_medals
                    FROM (
                    SELECT event_id, MIN(part_id) AS first_part_id
                    FROM athlete_part
                    GROUP BY event_id,medal
                    ) AS first_occurrence
                    JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
                    JOIN events ev ON ap.event_id = ev.event_id
                    JOIN sports sp ON ev.sport_id = sp.sport_id
                    JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
                    JOIN country_details cd ON cd.country_id = cp.country_id
        WHERE cp.edition <= '2020 Summer Olympics' AND ap.medal IS NOT NULL GROUP BY cp.as_country
        ORDER BY total_medals DESC LIMIT 5;
                """
    },
    {
        'input': 'Name the country who won most gold medals in Olympics',
        'query': """SELECT cd.country_name, count(ap.medal) AS total_medals
                FROM (
                SELECT event_id, MIN(part_id) AS first_part_id
                FROM athlete_part
                GROUP BY event_id,medal
                ) AS first_occurrence
                JOIN athlete_part ap ON first_occurrence.first_part_id = ap.part_id
                JOIN events ev ON ap.event_id = ev.event_id
                JOIN sports sp ON ev.sport_id = sp.sport_id
                JOIN country_participations cp ON cp.participation_id = sp.country_participation_id
                JOIN country_details cd ON cd.country_id = cp.country_id
                WHERE ap.medal = "Gold" GROUP BY cp.as_country
                ORDER BY total_medals DESC LIMIT 1;"""
    }
]
