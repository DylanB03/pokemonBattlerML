TASK: Select the listed legal action that maximizes eventual battle win probability.
RULES:
- Use only the information in BATTLE_STATE.
- Treat unknownitem, unknownability, notype, and omitted opponent moves as unknown.
- Consider long-term strategy, not only immediate damage.
- Output exactly one listed action ID and nothing else.

<BATTLE_STATE>
format=gen9ou
forced_switch=false
can_tera=true
weather=noweather
battle_field=nofield
player_conditions=stealthrock
opponent_conditions=spikes
opponents_remaining=5
opponent_teampreview=dragapult,gholdengo,gliscor,greattusk,kingambit,primarina
player_prev_move=knockoff
opponent_prev_move=ironhead

<PLAYER_ACTIVE_POKEMON>
name=greattusk
base_species=greattusk
hp_pct=0.61
types=fighting ground
tera_type=water
item=leftovers
ability=protosynthesis
lvl=100
status=nostatus
effect=noeffect
base_hp=115
base_atk=131
base_def=131
base_spa=53
base_spd=53
base_spe=87
atk_boost=1
def_boost=0
spa_boost=0
spd_boost=0
spe_boost=0
accuracy_boost=0
evasion_boost=0
moves:
- name=earthquake move_type=ground category=physical base_power=100 accuracy=1.0 priority=0 current_pp=12 max_pp=16
- name=icespinner move_type=ice category=physical base_power=80 accuracy=1.0 priority=0 current_pp=24 max_pp=24
- name=knockoff move_type=dark category=physical base_power=65 accuracy=1.0 priority=0 current_pp=31 max_pp=32
- name=rapidspin move_type=normal category=physical base_power=50 accuracy=1.0 priority=0 current_pp=64 max_pp=64
</PLAYER_ACTIVE_POKEMON>

<OPPONENT_ACTIVE_POKEMON>
name=kingambit
base_species=kingambit
hp_pct=0.74
types=dark steel
tera_type=notype
item=unknownitem
ability=supremeoverlord
lvl=100
status=brn
effect=noeffect
base_hp=100
base_atk=135
base_def=120
base_spa=60
base_spd=85
base_spe=50
atk_boost=0
def_boost=0
spa_boost=0
spd_boost=0
spe_boost=0
accuracy_boost=0
evasion_boost=0
moves:
- name=ironhead move_type=steel category=physical base_power=80 accuracy=1.0 priority=0 current_pp=23 max_pp=24
- name=suckerpunch move_type=dark category=physical base_power=70 accuracy=1.0 priority=1 current_pp=7 max_pp=8
</OPPONENT_ACTIVE_POKEMON>

<AVAILABLE_SWITCH action_id=A4>
name=corviknight
base_species=corviknight
hp_pct=0.76
types=flying steel
tera_type=dragon
item=leftovers
ability=pressure
lvl=100
status=nostatus
effect=noeffect
base_hp=98
base_atk=87
base_def=105
base_spa=53
base_spd=85
base_spe=67
atk_boost=0
def_boost=0
spa_boost=0
spd_boost=0
spe_boost=0
accuracy_boost=0
evasion_boost=0
moves:
- name=bodypress move_type=fighting category=physical base_power=80 accuracy=1.0 priority=0 current_pp=16 max_pp=16
- name=irondefense move_type=steel category=status base_power=0 accuracy=1.0 priority=0 current_pp=24 max_pp=24
- name=roost move_type=flying category=status base_power=0 accuracy=1.0 priority=0 current_pp=8 max_pp=8
- name=uturn move_type=bug category=physical base_power=70 accuracy=1.0 priority=0 current_pp=32 max_pp=32
</AVAILABLE_SWITCH>

<AVAILABLE_SWITCH action_id=A5>
name=dragapult
base_species=dragapult
hp_pct=0.88
types=dragon ghost
tera_type=ghost
item=choicespecs
ability=infiltrator
lvl=100
status=nostatus
effect=noeffect
base_hp=88
base_atk=120
base_def=75
base_spa=100
base_spd=75
base_spe=142
atk_boost=0
def_boost=0
spa_boost=0
spd_boost=0
spe_boost=0
accuracy_boost=0
evasion_boost=0
moves:
- name=dracometeor move_type=dragon category=special base_power=130 accuracy=0.9 priority=0 current_pp=8 max_pp=8
- name=fireblast move_type=fire category=special base_power=110 accuracy=0.85 priority=0 current_pp=8 max_pp=8
- name=shadowball move_type=ghost category=special base_power=80 accuracy=1.0 priority=0 current_pp=24 max_pp=24
- name=uturn move_type=bug category=physical base_power=70 accuracy=1.0 priority=0 current_pp=32 max_pp=32
</AVAILABLE_SWITCH>
</BATTLE_STATE>

<LEGAL_ACTIONS>
<A0> universal_action=0 type=move name=earthquake terastallize=false
<A1> universal_action=1 type=move name=icespinner terastallize=false
<A2> universal_action=2 type=move name=knockoff terastallize=false
<A3> universal_action=3 type=move name=rapidspin terastallize=false
<A4> universal_action=4 type=switch species=corviknight
<A5> universal_action=5 type=switch species=dragapult
<A9> universal_action=9 type=move name=earthquake terastallize=true
<A10> universal_action=10 type=move name=icespinner terastallize=true
<A11> universal_action=11 type=move name=knockoff terastallize=true
<A12> universal_action=12 type=move name=rapidspin terastallize=true
</LEGAL_ACTIONS>

<ACTION>
