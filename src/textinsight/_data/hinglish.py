"""Romanized Hindi word lists.

Romanized Hindi has no standard spelling, so one word appears in many forms.
``VARIANT_GROUPS`` lists a canonical spelling followed by its known variants.

Spellings that are also ordinary English words ("me", "or", "he") are left out
of the variant lists: mapping them would corrupt English text.
"""

from __future__ import annotations

VARIANT_GROUPS: tuple[str, ...] = (
    # Verbs and auxiliaries
    "hai haii hae",
    "hain hein hn",
    "tha thaa",
    "thi thii",
    "raha rha rahaa",
    "rahi rhi rahii",
    "rahe rhe",
    "gaya gya gayaa",
    "gayi gyi gayee",
    "gaye gye",
    "hua huaa",
    "hui huii",
    "hue huye",
    "hoga hogaa",
    "hogi hogii",
    "hota hotaa",
    "hoti hotii",
    "hote hotey",
    "karna krna karnaa",
    "karo kro",
    "kar kr",
    "karte krte kartey",
    "karta krta",
    "karti krti",
    "kiya",
    "kiye kie",
    "liya lia",
    "diya dia",
    "dena",
    "lena",
    "jana jaana",
    "aana",
    "aaya aya",
    "aayi ayi aai",
    "lagta lgta",
    "lagti lgti",
    "laga lga",
    "lagi lgi",
    "chahiye chahie chaiye chahiyee chahye chaheye chayiye",
    "sakta skta",
    "sakti skti",
    "dekha dekhaa",
    "dekho dkho",
    "bolo blo",
    "bola bolaa",
    "khareeda kharida",
    "mila milaa",
    "mili milii",
    "chal chl",
    "chalo chlo",
    "pata pta",
    "samajh smjh samjh",
    # Pronouns and determiners
    "main mai mae",
    "mujhe mjhe muje mujhey mujhy",
    "mera meraa",
    "meri merii",
    "mere merey",
    "tum tumm",
    "tumhe tumhein tumhay",
    "tumhara tumhaara",
    "tu tuu",
    "tera teraa",
    "teri terii",
    "tere terey",
    "aap aapp",
    "aapka apka",
    "aapki apki",
    "aapke apke",
    "hum humm",
    "humne hmne",
    "hamara humara hmara",
    "hamare humare hmare hamaare",
    "yeh ye yah yehh",
    "woh wo voh vo",
    "iska iskaa",
    "iski iskii",
    "iske iskey",
    "uska uskaa",
    "uski uskii",
    "uske uskey",
    "unka unkaa",
    "inka inkaa",
    "apna apnaa",
    "apni apnii",
    "apne apney",
    "koi koii",
    "kuch kuchh kch",
    "sab sb",
    "sabhi sbhi",
    "ek ekk",
    # Question words and connectors
    "kya kyaa",
    "kyun kyu kyon kyoon",
    "kaise kese kaisay",
    "kaisa kesa",
    "kaisi kesi",
    "kab kabb",
    "kahan kaha kahaan",
    "kaun kon",
    "aur aor",
    "lekin lkin lakin",
    "magar mgr",
    "kyunki kyuki kyonki kiunki",
    "isliye isiliye islye",
    "phir fir phr",
    "warna varna",
    "agar agr",
    "toh tho",
    "bhi bhii bhee",
    "sirf srf",
    "bas bass",
    "abhi abi abhii",
    "kabhi kbhi kabi",
    "aise aisay",
    "waise wese vaise",
    "yahan yaha yahaan",
    "wahan waha wahaan",
    # Postpositions
    "ka kaa",
    "ki kii",
    "ke",
    "ko koo",
    "se sey",
    "mein mien mei",
    "par pr",
    "pe pey",
    "tak tk",
    "liye liyee",
    "saath sath",
    "pehle pahle phle",
    "andar ander",
    "bahar baahar",
    "upar uppar",
    "neeche niche",
    # Negation and affirmation
    "nahi nahin nai nhi nahee nahii nahiin nehi nahe",
    "na naa",
    "mat matt",
    "haan han haa",
    "ji jii",
    # Intensifiers and quantity
    "bahut bohot bohat bahot bhot bhut boht bht bohut bahoot bahat",
    "zyada jyada zyaada jyaada",
    "thoda thora thodaa",
    "thodi thori",
    "kam kamm",
    "pura poora",
    "puri poori",
    "ekdum ekdam",
    "bilkul bilkull blkl",
    # Opinion words
    "accha acha achha acchaa achaa achchha aacha",
    "acchi achi achhi achii",
    "acche ache achhe",
    "bekaar bekar bakaar",
    "bakwas bakwaas bakvas bakvaas",
    "zabardast jabardast zabrdast jabrdast",
    "mast mst",
    "badhiya badiya badhia badia bdhiya",
    "sahi sahii shi",
    "galat ghalat glt",
    "thik theek thk thek",
    "bura buraa",
    "buri burii",
    "bure burey",
    "ganda gandaa",
    "gandi gandii",
    "khatarnak khatarnaak",
    "sundar sundr",
    "khubsurat khoobsurat khoobsoorat",
    "pyara pyaara",
    "pyari pyaari",
    "pyaar pyar piyar",
    "sasta saasta",
    "sasti saasti",
    "mehenga mehnga mahanga mahenga",
    "mehengi mehngi mahangi",
    "khush kush",
    "dukhi dukhii",
    "gussa ghussa gusa",
    "paagal pagal",
    "dhoka dhokha",
    "faltu faaltu",
    "kharab khraab kharaab",
    "barbaad barbad",
    "shandaar shandar",
    "behtareen behtarin",
    "ghatiya ghatia",
    "nirash niraash",
    # Everyday nouns
    "yaar yarr yaarr",
    "bhai bhaii bhaiya",
    "dost dosst",
    "log logg",
    "paisa pesa",
    "paise pese",
    "khana khaana",
    "paani pani",
    "ghar ghr",
    "kaam kaamm",
    "kafi kaafi",
    "baki baaki",
    "haath hath",
    "baat baatt",
    "matlab mtlb matlb",
    "zindagi jindagi zindgi",
    "dil dill",
    "maza mazaa maja",
    "jaldi jldi",
    "dhanyavaad dhanyawad dhanyavad",
    "shukriya sukriya shukria",
    "zaroor jaroor zarur jarur",
    "wala waala vala",
    "wali waali vali",
    "wale waale vale",
    "aaj aj",
    "kal kall",
    "din dinn",
    "raat raatt",
    "samay samey",
    "waqt wakt",
    "duniya dunia",
    "maa maaa",
    "beta betaa",
    "beti betii",
    "sach sachh",
    "jhooth jhoot jhut",
    "dikkat dikat",
    "pareshan pareshaan",
    "mushkil muskil",
    "aasan asan",
    "naya nayaa",
    "nayi",
    "purana puraana",
    "bada badaa",
    "badi badii",
    "chota chhota",
    "choti chhoti",
    "lamba lambaa",
)


def _parse() -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {}
    for line in VARIANT_GROUPS:
        canonical, *variants = line.split()
        groups[canonical] = tuple(variants)
    return groups


GROUPS: dict[str, tuple[str, ...]] = _parse()

STOPWORDS_HINGLISH = frozenset(
    """
    hai hain tha thi raha rahi rahe ka ki ke ko se mein par pe tak aur bhi toh yeh woh iska
    iski iske uska uski uske unka inka main mujhe mera meri mere tum tumhe tumhara tu tera
    teri tere aap aapka aapki aapke hum humne hamara kya koi kuch sab ek hoga hogi hota hoti
    hote liye saath ji
    """.split()
)

# Extra words that carry grammar or degree but do not name a topic. These are kept
# apart from STOPWORDS_HINGLISH because only keywords and summaries ignore them.
FUNCTION_WORDS_HINGLISH = frozenset(
    """
    bahut bohot kaafi zyada jyada thoda thodi thode kam itna itni itne utna utni utne
    lekin magar par ya agar phir fir abhi ab jab tab kab kyun kyu kaise kaisa kaisi kahan
    yahan wahan idhar udhar waha yaha jo jis jise jaise waise sirf bas hi nhi bilkul ekdum
    ho hua hui hue hun hoon hu kar karo karna karta karti karte karenge karega karegi kiya
    kiye ki gaya gayi gaye gya gyi aa aaya aayi aaye aata aati aate aana aayega aayegi
    ja jaa jana jata jati jate jayega jayegi le lo lena leta leti lete liya liye li
    de do dena deta deti dete diya diye di mil mila mili mile milta milti milte
    chal chala chali chale chalta chalti chalte dekh dekha dekhi dekhna sakta sakti sakte
    chahiye wala wali wale apna apni apne unke unki inke inki kis kisi kuchh
    yaar bhai dost
    """.split()
)
