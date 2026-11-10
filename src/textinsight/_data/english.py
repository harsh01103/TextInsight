"""Common English words and stopwords."""

from __future__ import annotations

# Negations are left out on purpose: dropping "not" flips the meaning of a sentence.
STOPWORDS_EN = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is
    it its itself just me more most my myself of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself yourselves
    """.split()
)

ENGLISH_COMMON = frozenset(
    """
    a about above absolutely across actually add after again against ago agree air all allow
    almost alone along already also although always am amazing among an and angry animal
    annoying another answer any anyone anything app apple apps are area arm around arrive art
    as ask at attention available average awesome away awful baby back bad bag bank battery
    be beautiful because become bed been before began begin behind being believe below best
    better between big bill black blue boat body book boring both box boy brand break bring
    broke broken brother brought build building bus business busy but buy by call came camera
    can cannot car card care carry case cause change charge charger cheap check child
    children choose city class clean clear close clothes coffee cold college color come
    comfortable company complete computer confused consider contact continue cool cost could
    country couple course cover create crazy customer cut dark data date daughter day days
    dead deal dear decide delivered delivery design did die different difficult dinner
    disappointed disappointing do doctor does doing dog dollar done door down dream drink
    drive drop during each early earth easy eat economy education effect either else email
    end enjoy enough entire even evening ever every everyone everything exactly example
    excellent except excited expect expensive experience explain eye face fact fail failed
    fair fake fall family fan fantastic far fast father fault favorite feature features feel
    feeling few field fight figure fill film final finally find fine finish first fit five
    fix flight floor follow food foot for force forget form forward found four free fresh
    friend friends from front full fun funny future game garden gave get getting girl give
    given glad go goes going gone good got government great green ground group grow guess guy
    guys had hair half hand happen happened happy hard has hate have having he head health
    hear heard heart heavy hello help her here high him himself his history hit hold holiday
    home hope horrible hospital hot hotel hour hours house how however huge human hundred
    hungry hurt husband i idea if ignore important improve in include including increase
    indeed information inside instead interest interesting internet into is issue issues it
    item its itself job join just keep kept key kid kids kind kitchen knew know known lack
    laptop large last late later laugh law lay lead learn least leave left less let letter
    level life light like likely line list listen little live living local long look looking
    lose lost lot lots love lovely low lunch machine made main major make making man manager
    many market marriage matter may maybe me mean meant meet meeting member men message met
    might mind minute minutes miss mobile model moment money month months more morning most
    mother move movie movies much music must my myself name nation near nearly need needed
    never new news next nice night nine no nobody none nor not note nothing notice now number
    of off offer office often oh ok okay old on once one online only open opinion option or
    order other others our out outside over own owner page paid pain paper parents park part
    particular party pass past pay payment people perfect perhaps person personal phone pick
    picture piece place plan play player please point police policy poor popular possible
    power prefer present president pretty price probably problem problems process product
    products program promise proper provide public pull purchase purchased put quality
    question quick quickly quite rain raise ran rather reach read ready real really reason
    receive received recent recommend red refund refused relationship remember remove repair
    replace replacement reply report rest restaurant result return review reviews rich ride
    right road room rude rule run sad safe said sale same saturday save saw say says scared
    school screen sea season seat second see seem seen sell send sense sent series serious
    service set seven several shall share she ship shipping shirt shop short should show
    shows side sign simple since sing sister sit site situation six size sleep slow small
    smart smell smile so social software some someone something sometimes son song soon sorry
    sound space speak special speed spend spent sport staff stand standard star start started
    state station stay still stop store story street strong student study stuff stupid style
    subject such suddenly summer sunday support suppose sure surprised sweet system table
    take taken talk taste tea teach teacher team tell ten terrible test than thank thanks
    that the their them themselves then there therefore these they thing things think third
    this those though thought thousand three through throughout thus ticket till time tired
    to today together told tomorrow tonight too took top total touch towards town traffic
    train travel tried trip trouble true trust truth try trying turn twenty two type under
    understand unfortunately unit university until up update upon us use used useful useless
    user using usually value very video view village visit voice wait waiting walk wall want
    wanted war warm warranty was wash waste watch water way we wear weather website week
    weekend weeks weight welcome well went were west what whatever when where whether which
    while white who whole why wife will win window wish with within without woman women won
    wonder wonderful word work worked working world worried worse worst worth would write
    writing wrong yeah year years yellow yes yesterday yet you young your yourself
    don't doesn't didn't won't can't isn't aren't wasn't weren't haven't hasn't hadn't
    wouldn't shouldn't couldn't i'm i've i'll i'd you're you've you'll they're they've
    we're we've it's that's there's what's let's he's she's
    """.split()
)

# Common verbs, quantities and research-writing words that rarely name a topic.
# Keywords and summaries ignore them; removing them is not safe for every task.
GENERIC_WORDS_EN = frozenset(
    """
    using used use uses based given shown show shows make makes made new high low large
    small different various several many much first second third one two three four five
    also may can could would however thus paper present presents presented proposed
    propose proposes results result study studies approach method methods problem problems
    """.split()
)

# Nouns ending in "-ly", which would otherwise be mistaken for adverbs.
LY_NOUNS = frozenset(
    "family assembly supply reply italy july anomaly monopoly rally ally belly jelly".split()
)
